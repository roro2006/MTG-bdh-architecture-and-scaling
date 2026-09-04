"""The contract between the training CLI and the Colab scaffolding.

`scripts/colab_run.sh` drives a long cell as a sequence of bounded segments:
exec the bootstrap, pull the artefacts, and go round again while there is more
to do. "More to do" is carried by a single number -- exit code 75 -- and the
resume state it continues from is two filenames. Neither is enforced by the
type system, and all three live in different files and different languages.

If `EXIT_INCOMPLETE` were changed to 76, nothing would fail to import and no
test elsewhere would notice. The driver would read a completed segment as a
crash, or worse, read an incomplete one as finished and report a truncated
curve as a result. So the agreement is asserted here rather than assumed.

The bootstrap cannot be imported to read its constants: it ends in a
module-level `sys.exit(main())`, deliberately, because `colab exec -f` does not
guarantee `__name__ == "__main__"`. Its constants are therefore read out of the
source with `ast`, which is also what keeps this test from executing it.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.training.checkpoint import RESUME_PARAMS_FILE, RESUME_STATE_FILE
from src.training.run import EXIT_INCOMPLETE

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
BOOTSTRAP = SCRIPTS / "colab_bootstrap.py"
DRIVER = SCRIPTS / "colab_run.sh"


def _literal(path: Path, name: str):
    """The value of a module-level constant, without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} defines no module-level {name}")


def test_scaffolding_files_are_present():
    assert BOOTSTRAP.is_file(), "the remote entry point is missing"
    assert DRIVER.is_file(), "the WSL driver is missing"


def test_bootstrap_agrees_with_run_py_on_the_incomplete_exit_code():
    """75 is the whole signal that a segmented run has more to do."""
    assert EXIT_INCOMPLETE == 75
    assert _literal(BOOTSTRAP, "EXIT_INCOMPLETE") == EXIT_INCOMPLETE


def test_driver_loops_on_the_same_exit_code():
    """The shell side carries the number as a literal; it must be the same one."""
    assert str(EXIT_INCOMPLETE) in DRIVER.read_text(encoding="utf-8")


def test_bootstrap_agrees_with_checkpoint_py_on_the_resume_filenames():
    """The bootstrap decides what to mirror off the VM by name."""
    assert set(_literal(BOOTSTRAP, "RESUME_FILES")) == {
        RESUME_PARAMS_FILE,
        RESUME_STATE_FILE,
    }


def test_bootstrap_runs_main_unconditionally():
    """`colab exec -f` does not promise __name__ == '__main__'.

    A bootstrap guarded by the usual `if __name__ == "__main__"` would be
    shipped to the kernel, define its functions, do nothing at all, and exit
    zero -- which the driver would read as a completed segment.
    """
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    calls_main = [
        node for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "exit"
    ]
    assert calls_main, "the bootstrap never calls main() at module level"


def test_bootstrap_is_stdlib_only_at_module_scope():
    """It runs before requirements are installed, and before jax is checked.

    A module-scope `import jax` here would fail on a bare runtime, or -- worse
    -- succeed and pin the pre-install version into the process that has to
    detect the post-install clobber.
    """
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    third_party = {"jax", "jaxlib", "flax", "optax", "numpy", "pandas", "scipy"}
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module.split(".")[0]]
        offending = sorted(set(names) & third_party)
        assert not offending, f"module-scope import of {offending} in the bootstrap"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_driver_is_syntactically_valid():
    result = subprocess.run(
        ["bash", "-n", str(DRIVER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _driver_flags_sent_to_bootstrap() -> set[str]:
    """The `--flags` colab_run.sh puts into the bootstrap's argv.

    Read out of `build_remote_script`, which is the one place the driver
    composes that argv.
    """
    text = DRIVER.read_text(encoding="utf-8")
    start = text.index("build_remote_script()")
    end = text.index("\n}", start)
    body = text[start:end]
    return set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", body))


def _bootstrap_accepted_flags() -> set[str]:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    return set(re.findall(r"add_argument\(\s*\"(--[a-z0-9-]+)\"", source))


def test_every_flag_the_driver_sends_is_one_the_bootstrap_accepts():
    """The failure this prevents costs a provision to discover.

    argparse rejects an unknown flag with exit 2 before `main()` runs, so a
    name that exists on only one side does not fail here -- it fails on the
    remote kernel, after a VM has been allocated, the repo cloned, the data
    staged and a segment's training done. The driver passes --resume on every
    segment after the first, so a mismatch there would specifically break the
    *second* iteration of the loop, with the first segment's work already on a
    VM that is about to be torn down.
    """
    sent = _driver_flags_sent_to_bootstrap()
    accepted = _bootstrap_accepted_flags()
    # Flags the driver builds for itself, not for the bootstrap's argparse.
    sent -= {"--"}
    unknown = sorted(sent - accepted)
    assert not unknown, (
        f"colab_run.sh sends {unknown} to colab_bootstrap.py, which does not "
        f"accept them. Accepted: {sorted(accepted)}"
    )


def test_bootstrap_resumes_by_default_and_can_be_told_not_to():
    """The segment loop depends on resume being on without being asked for."""
    module = _load_bootstrap_without_running_it()
    base = ["--sets", "FIN", "--train-set", "FIN"]

    assert module["parse_args"](base).resume is True
    assert module["parse_args"](base + ["--resume"]).resume is True
    assert module["parse_args"](base + ["--no-resume"]).resume is False


def _load_bootstrap_without_running_it() -> dict:
    """Executes the bootstrap's definitions, minus its module-level exit."""
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "attr", None) == "exit"
        )
    ]
    namespace: dict = {}
    exec(compile(tree, str(BOOTSTRAP), "exec"), namespace)
    return namespace


def test_resume_flag_reaches_run_py_only_when_asked():
    module = _load_bootstrap_without_running_it()
    base = ["--sets", "FIN", "--train-set", "FIN"]
    build = module["build_training_argv"]

    on = build(module["parse_args"](base), Path("/p"), Path("/o"))
    off = build(module["parse_args"](base + ["--no-resume"]), Path("/p"), Path("/o"))
    assert "--resume" in on
    assert "--resume" not in off


def test_default_run_name_separates_cells_sized_by_epochs():
    """The run name is the artefact directory, and that is where resume lives.

    `--epochs` overrides `--steps` inside run.py, so two grid cells at 1 and 3
    epochs both carry the default 3000 if the name reads `--steps`. They then
    share an artefact directory, and the second does not overwrite the first --
    it finds the first's resume state and *continues* it, reporting a
    plausible number for an experiment nobody ran. That is a wrong result
    rather than a missing one, which is why it is asserted here.
    """
    module = _load_bootstrap_without_running_it()
    parse_args, default_run_name = module["parse_args"], module["default_run_name"]

    def cell(*extra):
        args = parse_args(["--sets", "FIN", "--train-set", "FIN", *extra])
        return default_run_name(args, "FIN")

    arm = ["--arm", "bdh"]

    # Sized by steps: unchanged from before --epochs existed.
    assert cell(*arm) == "bdh_d64_FIN_s3000"

    # Sized by epochs: named by epochs, and two epoch counts differ.
    assert cell(*arm, "--epochs", "3") == "bdh_d64_FIN_e3"
    assert cell(*arm, "--epochs", "1") != cell(*arm, "--epochs", "3")

    # The same epoch count at a different data fraction is a different cell,
    # which is the whole D axis.
    assert cell(*arm, "--epochs", "3", "--data-fraction", "0.25") != cell(
        *arm, "--epochs", "3"
    )

    # And so is the same cell at a different neuron multiplier, now that
    # PROJECT_PLAN section 4 has unpinned it.
    assert cell(*arm, "--epochs", "3", "--neuron-multiplier", "8") != cell(
        *arm, "--epochs", "3"
    )
    assert cell(*arm, "--neuron-multiplier", "4") == cell(*arm)


def test_the_driver_forwards_the_grid_flags_and_names_by_them():
    """--epochs is unusable for a grid if the driver cannot express it.

    Section 6 requires the D axis to be data scale, which requires cells to
    be sized in epochs. The driver had `--steps` only, so every grid cell had
    to reach `--epochs` through the `--` passthrough while the run name went
    on reading `--steps` -- the collision above, arriving by a different road.
    """
    source = DRIVER.read_text(encoding="utf-8")

    for flag, var in (
        ("--epochs", "EPOCHS"),
        ("--data-fraction", "DATA_FRACTION"),
        ("--neuron-multiplier", "NEURON_MULTIPLIER"),
    ):
        assert f"{flag})" in source, f"the driver does not accept {flag}"
        assert f'bootstrap_argv+=({flag} "${var}")' in source.replace("  ", " "), (
            f"the driver accepts {flag} but never forwards it, which is worse "
            "than not accepting it: the cell would train at the default and "
            "report the flag in its plan"
        )

    # The default name has to branch on how the cell was sized.
    assert 'RUN_NAME="${ARM}_d${WIDTH}_e${EPOCHS}"' in source
