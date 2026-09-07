#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text


console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "HF Model Quantizer"

DEFAULT_LLAMA_DIR = Path("./llama.cpp")
DEFAULT_MODELS_DIR = Path("./models")


QUANT_OPTIONS = {
    "1": {
        "name": "Q8_0",
        "description": "Very high quality, large size",
        "group": "Standard",
    },
    "2": {
        "name": "Q6_K",
        "description": "Excellent quality",
        "group": "Standard",
    },
    "3": {
        "name": "Q5_K_M",
        "description": "High quality / good compression",
        "group": "Standard",
    },
    "4": {
        "name": "Q4_K_M",
        "description": "Recommended general-purpose Q4",
        "group": "Standard",
    },
    "5": {
        "name": "Q4_K_S",
        "description": "Smaller Q4 variant",
        "group": "Standard",
    },
    "6": {
        "name": "Q4_0",
        "description": "Classic Q4 format",
        "group": "Standard",
    },
    "7": {
        "name": "Q3_K_M",
        "description": "Aggressive compression",
        "group": "Low-bit",
    },
    "8": {
        "name": "Q3_K_S",
        "description": "Smaller Q3 variant",
        "group": "Low-bit",
    },
    "9": {
        "name": "Q2_K",
        "description": "Very aggressive compression",
        "group": "Ultra-low-bit",
    },
    "10": {
        "name": "IQ4_XS",
        "description": "Importance-aware Q4",
        "group": "IQ",
    },
    "11": {
        "name": "IQ3_M",
        "description": "Importance-aware Q3",
        "group": "IQ",
    },
    "12": {
        "name": "IQ2_M",
        "description": "Importance-aware ultra-low-bit",
        "group": "IQ",
    },
    "13": {
        "name": "IQ2_XS",
        "description": "Smaller IQ2 variant",
        "group": "IQ",
    },
    "14": {
        "name": "IQ2_XXS",
        "description": "Extremely aggressive IQ2",
        "group": "Experimental",
    },
    "15": {
        "name": "IQ1_M",
        "description": "Extreme compression; substantial quality loss possible",
        "group": "Experimental",
    },
    "16": {
        "name": "IQ1_S",
        "description": "Smallest IQ1-style option",
        "group": "Experimental",
    },
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def header() -> None:
    title = Text()
    title.append("HF MODEL ", style="bold cyan")
    title.append("QUANTIZER", style="bold magenta")

    console.print(
        Panel(
            title,
            subtitle="Hugging Face → GGUF → llama.cpp quantization",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def print_step(title: str, description: str | None = None) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")

    if description:
        console.print(f"[dim]{description}[/dim]")


def print_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def human_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PiB"


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


def run(command: list[str]) -> None:
    console.print()
    console.print(
        "[dim]$ " + " ".join(str(x) for x in command) + "[/dim]"
    )
    console.print()

    subprocess.run(command, check=True)


# ---------------------------------------------------------------------------
# llama.cpp discovery/build
# ---------------------------------------------------------------------------


def find_quantizer(llama_dir: Path) -> Path | None:
    candidates = [
        llama_dir / "build/bin/llama-quantize",
        llama_dir / "build/bin/Release/llama-quantize",
        llama_dir / "llama-quantize",
    ]

    if os.name == "nt":
        candidates += [
            llama_dir / "build/bin/Release/llama-quantize.exe",
            llama_dir / "build/bin/llama-quantize.exe",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def ensure_llama_cpp(llama_dir: Path) -> Path:
    if not llama_dir.exists():
        print_warning("llama.cpp was not found.")

        clone = Confirm.ask(
            "Clone llama.cpp automatically?",
            default=True,
        )

        if not clone:
            raise RuntimeError(
                "llama.cpp is required for GGUF conversion and quantization."
            )

        run(
            [
                "git",
                "clone",
                "https://github.com/ggml-org/llama.cpp.git",
                str(llama_dir),
            ]
        )

    converter = llama_dir / "convert_hf_to_gguf.py"

    if not converter.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py was not found in:\n{llama_dir}"
        )

    quantizer = find_quantizer(llama_dir)

    if quantizer:
        print_success(f"Found llama-quantize: {quantizer}")
        return quantizer

    print_warning("llama.cpp exists but llama-quantize is not built.")

    build = Confirm.ask(
        "Build llama.cpp now?",
        default=True,
    )

    if not build:
        raise RuntimeError("llama-quantize is required.")

    print_step(
        "Building llama.cpp",
        "This is normally required only once.",
    )

    run(
        [
            "cmake",
            "-S",
            str(llama_dir),
            "-B",
            str(llama_dir / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )

    run(
        [
            "cmake",
            "--build",
            str(llama_dir / "build"),
            "--config",
            "Release",
            "-j",
        ]
    )

    quantizer = find_quantizer(llama_dir)

    if not quantizer:
        raise RuntimeError(
            "Build completed, but llama-quantize could not be found."
        )

    return quantizer


# ---------------------------------------------------------------------------
# Hugging Face model handling
# ---------------------------------------------------------------------------


def model_directory_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def resolve_hf_model(
    model_input: str,
    models_dir: Path,
) -> Path:
    local_path = Path(model_input).expanduser()

    if local_path.exists():
        if not local_path.is_dir():
            raise RuntimeError(
                "The supplied local model path is not a directory."
            )

        print_success(
            f"Using local Hugging Face model: {local_path.resolve()}"
        )

        return local_path.resolve()

    model_name = model_directory_name(model_input)
    destination = models_dir / "hf" / model_name

    destination.parent.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")

    if destination.exists():
        console.print()
        console.print(
            Panel(
                f"[bold]{destination}[/bold]\n\n"
                "A local copy of this model already exists.",
                title="Existing model",
                border_style="yellow",
            )
        )

        reuse = Confirm.ask(
            "Reuse the existing download?",
            default=True,
        )

        if reuse:
            return destination.resolve()

    print_step(
        "Downloading model",
        model_input,
    )

    snapshot_download(
        repo_id=model_input,
        local_dir=destination,
        token=token,
    )

    print_success("Hugging Face model downloaded.")

    return destination.resolve()


# ---------------------------------------------------------------------------
# Quantization menu
# ---------------------------------------------------------------------------


def show_quant_menu() -> None:
    table = Table(
        title="Available Quantization Formats",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("#", justify="right", style="bold")
    table.add_column("Format", style="bold green")
    table.add_column("Class")
    table.add_column("Description")

    for key, item in QUANT_OPTIONS.items():
        table.add_row(
            key,
            item["name"],
            item["group"],
            item["description"],
        )

    console.print(table)


def choose_quantizations() -> list[str]:
    show_quant_menu()

    console.print()
    console.print(
        "[dim]"
        "Enter one number, multiple comma-separated numbers, "
        "or 'all'."
        "[/dim]"
    )

    while True:
        choice = Prompt.ask(
            "Quantization",
            default="4",
        ).strip().lower()

        if choice == "all":
            return [
                item["name"]
                for item in QUANT_OPTIONS.values()
            ]

        selected = []

        try:
            keys = [
                value.strip()
                for value in choice.split(",")
            ]

            for key in keys:
                if key not in QUANT_OPTIONS:
                    raise ValueError

                quant = QUANT_OPTIONS[key]["name"]

                if quant not in selected:
                    selected.append(quant)

            return selected

        except ValueError:
            print_error("Invalid selection. Try again.")


# ---------------------------------------------------------------------------
# GGUF conversion
# ---------------------------------------------------------------------------


def convert_to_gguf(
    model_dir: Path,
    model_name: str,
    output_dir: Path,
    llama_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16_file = output_dir / f"{model_name}-BF16.gguf"

    if bf16_file.exists():
        console.print()
        print_success(
            f"Existing BF16 GGUF found: {bf16_file}"
        )

        if Confirm.ask(
            "Reuse this BF16 GGUF?",
            default=True,
        ):
            return bf16_file

    converter = llama_dir / "convert_hf_to_gguf.py"

    print_step(
        "Converting Hugging Face model → GGUF",
        "Generating the high-precision BF16 source GGUF.",
    )

    run(
        [
            sys.executable,
            str(converter),
            str(model_dir),
            "--outfile",
            str(bf16_file),
            "--outtype",
            "bf16",
        ]
    )

    print_success("BF16 GGUF created.")

    return bf16_file


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_model(
    quantizer: Path,
    source: Path,
    destination: Path,
    quant_type: str,
) -> None:
    if destination.exists():
        print_warning(
            f"{destination.name} already exists."
        )

        overwrite = Confirm.ask(
            "Overwrite it?",
            default=False,
        )

        if not overwrite:
            print_success(
                f"Keeping existing {quant_type} model."
            )
            return

        destination.unlink()

    print_step(
        f"Quantizing → {quant_type}",
        destination.name,
    )

    run(
        [
            str(quantizer),
            str(source),
            str(destination),
            quant_type,
        ]
    )

    print_success(f"{quant_type} completed.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def show_summary(
    model_input: str,
    model_dir: Path,
    output_dir: Path,
    quantizations: list[str],
) -> None:
    table = Table(
        title="Quantization Results",
        box=box.ROUNDED,
        header_style="bold cyan",
    )

    table.add_column("Format", style="bold green")
    table.add_column("Size", justify="right")
    table.add_column("Output")

    model_name = model_dir.name

    for quant in quantizations:
        path = output_dir / f"{model_name}-{quant}.gguf"

        if path.exists():
            size = human_size(path.stat().st_size)

            table.add_row(
                quant,
                size,
                str(path),
            )

    console.print()
    console.print(table)

    console.print()

    console.print(
        Panel(
            f"[bold green]Quantization complete![/bold green]\n\n"
            f"[bold]Input:[/bold]   {model_input}\n"
            f"[bold]Output:[/bold]  {output_dir}",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    clear_screen()
    header()

    console.print(
        "Convert Hugging Face models to GGUF and generate "
        "multiple llama.cpp quantizations.\n"
    )

    model_input = Prompt.ask(
        "[bold]Hugging Face model ID or local HF directory[/bold]",
        default="google/gemma-3-4b-it",
    ).strip()

    models_dir = DEFAULT_MODELS_DIR.resolve()
    llama_dir = DEFAULT_LLAMA_DIR.resolve()

    console.print()

    try:
        quantizer = ensure_llama_cpp(llama_dir)

        model_dir = resolve_hf_model(
            model_input=model_input,
            models_dir=models_dir,
        )

        model_name = model_dir.name

        output_dir = (
            models_dir
            / "gguf"
            / model_name
        )

        console.print()
        quantizations = choose_quantizations()

        console.print()

        selection_table = Table(
            title="Selected Formats",
            box=box.SIMPLE,
        )

        selection_table.add_column("Format")

        for quant in quantizations:
            selection_table.add_row(quant)

        console.print(selection_table)

        if not Confirm.ask(
            "Start conversion and quantization?",
            default=True,
        ):
            console.print("[yellow]Cancelled.[/yellow]")
            return

        bf16_file = convert_to_gguf(
            model_dir=model_dir,
            model_name=model_name,
            output_dir=output_dir,
            llama_dir=llama_dir,
        )

        successful = []

        for quant in quantizations:
            destination = (
                output_dir
                / f"{model_name}-{quant}.gguf"
            )

            try:
                quantize_model(
                    quantizer=quantizer,
                    source=bf16_file,
                    destination=destination,
                    quant_type=quant,
                )

                if destination.exists():
                    successful.append(quant)

            except subprocess.CalledProcessError:
                print_error(
                    f"{quant} failed or is unsupported "
                    "for this model/build."
                )

                continue

        console.print()

        keep_bf16 = Confirm.ask(
            "Keep the intermediate BF16 GGUF?",
            default=True,
        )

        if not keep_bf16 and bf16_file.exists():
            bf16_file.unlink()
            print_success("Intermediate BF16 GGUF deleted.")

        show_summary(
            model_input=model_input,
            model_dir=model_dir,
            output_dir=output_dir,
            quantizations=successful,
        )

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Interrupted by user.[/yellow]")

    except subprocess.CalledProcessError as exc:
        console.print()
        print_error(
            f"Command failed with exit code {exc.returncode}."
        )

        sys.exit(exc.returncode)

    except Exception as exc:
        console.print()
        print_error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
