import typer

from PyDockStats.pydockstats import NAME, PyDockStats


app = typer.Typer(
    name=NAME,
    help=(
        "PyDockStats builds ROC (Receiver Operating Characteristic) and "
        "Predictiveness Curve plots for virtual screening programs."
    ),
)


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Data file", show_default=False),
    programs: str = typer.Option(
        "",
        "--programs",
        "-p",
        help="Programs names separated by comma",
    ),
    output: str = typer.Option("out.png", "--output", "-o", help="Output image name"),
    model: str = typer.Option(
        "logistic_regression",
        "--model",
        "-m",
        help="Model type (logistic regression only)",
    ),
) -> None:
    pydock = PyDockStats(programs, model, output)

    # preprocess the data
    scores, actives = pydock.preprocess(file)

    # calculate the PC and the ROC
    pydock.generate_plots(scores, actives)


def main() -> int:
    app()
    return 0
