#/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import aiohttp
import click

from models import Config
from mtg_card_renderer_mtgrender import MTGCardRendererMtgrender

# Pour le PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from PIL import Image


# ---------- Modèle interne de carte ----------

@dataclass
class DeckEntry:
    name: str
    set_code: str
    collector_number: str


# ---------- Parsing decklist Moxfield texte (mode historique) ----------

# 1 <name> (<SET>) <collector_number> [*F*]
LINE_RE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+\((\w+)\)\s+(\S+)(?:\s+\*F\*)?\s*$"
)


def parse_decklist_line(line: str) -> Optional[DeckEntry]:
    m = LINE_RE.match(line)
    if not m:
        return None
    count_str, name, set_code, num = m.groups()
    try:
        count = int(count_str)
    except ValueError:
        return None
    if count <= 0:
        return None

    return DeckEntry(
        name=name.strip(),
        set_code=set_code.strip(),
        collector_number=num.strip(),
    )


def parse_text_decklist_file(path: Path) -> List[DeckEntry]:
    text = path.read_text(encoding="utf-8")
    entries: List[DeckEntry] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        entry = parse_decklist_line(raw)
        if entry:
            entries.append(entry)
        else:
            click.echo(f"⚠️ Ligne non reconnue dans le fichier texte : {raw}", err=True)
    return entries


# ---------- Parsing Moxfield API v2 (mode URL) ----------

def extract_moxfield_deck_id(url: str) -> str:
    base = url.split("?", 1)[0]
    parts = base.rstrip("/").split("/")
    if not parts:
        raise ValueError(f"URL Moxfield invalide : {url}")
    deck_id = parts[-1]
    if not deck_id:
        raise ValueError(f"Impossible d'extraire l'ID de deck de l'URL : {url}")
    return deck_id


async def fetch_moxfield_deck_json(deck_url: str) -> Dict[str, Any]:
    deck_id = extract_moxfield_deck_id(deck_url)
    api_url = f"https://api2.moxfield.com/v2/decks/all/{deck_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Erreur en récupérant le deck Moxfield ({resp.status}) : {text[:200]}..."
                )
            return await resp.json()


def parse_moxfield_section(section: Dict[str, Any]) -> List[DeckEntry]:
    entries: List[DeckEntry] = []
    if not isinstance(section, dict):
        return entries

    for key, obj in section.items():
        if not isinstance(obj, dict):
            click.echo(
                f"⚠️ Entrée inattendue dans la section Moxfield (clé={key}, type={type(obj)})",
                err=True,
            )
            continue
        card_info = obj.get("card")
        if not isinstance(card_info, dict):
            click.echo(
                f"⚠️ Pas de champ 'card' exploitable dans la section Moxfield (clé={key})",
                err=True,
            )
            continue

        try:
            name = card_info["name"]
            set_code = card_info["set"]
            cn = card_info["cn"]
        except KeyError as e:
            click.echo(
                f"⚠️ Champ manquant dans 'card' pour la clé={key} : {e}",
                err=True,
            )
            continue

        collector_number = str(cn).strip()
        entries.append(DeckEntry(name=name, set_code=set_code, collector_number=collector_number))

    return entries


def parse_moxfield_json_to_entries(data: Dict[str, Any]) -> List[DeckEntry]:
    entries: List[DeckEntry] = []

    mainboard = data.get("mainboard")
    if isinstance(mainboard, dict):
        mb_entries = parse_moxfield_section(mainboard)
        click.echo(f"Section mainboard : {len(mb_entries)} cartes")
        entries.extend(mb_entries)
    else:
        click.echo("⚠️ Pas de section 'mainboard' exploitable dans le JSON Moxfield.", err=True)

    sideboard = data.get("sideboard")
    if isinstance(sideboard, dict) and sideboard:
        sb_entries = parse_moxfield_section(sideboard)
        click.echo(f"Section sideboard : {len(sb_entries)} cartes")
        entries.extend(sb_entries)

    return entries


# ---------- Scryfall (async) ----------

async def fetch_scryfall_card(
    session: aiohttp.ClientSession,
    entry: DeckEntry,
) -> Optional[dict]:
    url = f"https://api.scryfall.com/cards/{entry.set_code.lower()}/{entry.collector_number}"
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                click.echo(
                    f"❌ Scryfall {entry.name} ({entry.set_code} {entry.collector_number}) "
                    f"-> {resp.status}: {text[:200]}...",
                    err=True,
                )
                return None
            data = await resp.json()
            if data.get("object") == "error":
                click.echo(
                    f"❌ Scryfall error for {entry.name} "
                    f"({entry.set_code} {entry.collector_number}): {data.get('details')}",
                    err=True,
                )
                return None
            click.echo(f"✅ Scryfall OK: {entry.name} ({entry.set_code} {entry.collector_number})")
            return data
    except Exception as e:
        click.echo(
            f"❌ Exception Scryfall {entry.name} "
            f"({entry.set_code} {entry.collector_number}): {e}",
            err=True,
        )
        return None


# ---------- Helpers couleurs / type / layout ----------

WUBRG_ORDER = ["W", "U", "B", "R", "G"]


def sort_wubrg(chars: List[str]) -> List[str]:
    order = {c: i for i, c in enumerate(WUBRG_ORDER)}
    return sorted(chars, key=lambda c: order.get(c, 99))


def compute_colors_field(s: dict, type_line_raw: str) -> List[str]:
    colors = s.get("colors") or []
    type_line = type_line_raw or ""

    if len(colors) == 0:
        if "Artifact" in type_line:
            return ["Artifact"]
        if "Land" in type_line:
            return ["Land"]
        return []

    if len(colors) == 1:
        return [colors[0]]

    sorted_cols = sort_wubrg(colors)
    joined = "".join(sorted_cols)
    return [joined]


def compute_layout(s: dict) -> str:
    layout = s.get("layout", "normal")
    if layout not in {"normal", "saga", "planeswalker"}:
        return "normal"
    return layout


def map_scryfall_to_render_json(s: dict) -> dict:
    if "card_faces" in s and s["card_faces"]:
        front = s["card_faces"][0]
        name = front.get("name", s.get("name", ""))
        mana_cost = front.get("mana_cost", s.get("mana_cost", ""))
        type_line = front.get("type_line", s.get("type_line", ""))
        oracle_text = front.get("oracle_text", s.get("oracle_text", "")) or ""
        image_uris = front.get("image_uris") or s.get("image_uris") or {}
    else:
        name = s.get("name", "")
        mana_cost = s.get("mana_cost", "")
        type_line = s.get("type_line", "")
        oracle_text = s.get("oracle_text", "") or ""
        image_uris = s.get("image_uris") or {}

    type_line = type_line.replace("—", "-")
    is_legendary = "Legendary" in type_line

    colors_field = compute_colors_field(s, type_line)

    art_crop = image_uris.get("art_crop")

    layout = compute_layout(s)
    set_code = s.get("set", "").lower()

    render: Dict[str, Any] = {
        "name": name,
        "layout": layout,
        "collector_number": s.get("collector_number", ""),
        "image_uris": {
            "art_crop": art_crop
        },
        "mana_cost": mana_cost,
        "type_line": type_line,
        "oracle_text": oracle_text,
        "colors": colors_field,
        "set": set_code,
        "rarity": s.get("rarity", "").lower(),
        "artist": s.get("artist", "Unknown"),
    }

    if is_legendary:
        render["legendary"] = True

    power = s.get("power")
    toughness = s.get("toughness")

    if power is None and "card_faces" in s and s["card_faces"]:
        power = s["card_faces"][0].get("power")
        toughness = s["card_faces"][0].get("toughness")

    if power is not None:
        render["power"] = str(power)
    if toughness is not None:
        render["toughness"] = str(toughness)

    loyalty = s.get("loyalty")
    if loyalty is None and "card_faces" in s and s["card_faces"]:
        loyalty = s["card_faces"][0].get("loyalty")
    if loyalty is not None:
        render["loyalty"] = str(loyalty)

    if s.get("flavor_text"):
        render["flavor_text"] = s["flavor_text"]
    elif "card_faces" in s and s["card_faces"]:
        ft = s["card_faces"][0].get("flavor_text")
        if ft:
            render["flavor_text"] = ft

    return render


# ---------- PDF : imposition 3x3 cartes 63x88mm ----------

def generate_pdf_from_images(
    images_dir: Path,
    pdf_path: Path,
    page_size=A4,
    cards_per_row: int = 3,
    cards_per_col: int = 3,
    card_width_mm: float = 63.0,
    card_height_mm: float = 88.0,
    margin_mm: float = 5.0,
) -> None:
    images = sorted(
        [p for p in images_dir.glob("*.png") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if not images:
        click.echo(f"⚠️ Aucun PNG trouvé dans {images_dir}, PDF non généré.", err=True)
        return

    click.echo(f"Génération du PDF {pdf_path} à partir de {len(images)} images...")

    page_w, page_h = page_size
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)

    card_w = card_width_mm * mm
    card_h = card_height_mm * mm
    margin = margin_mm * mm

    total_cards_w = cards_per_row * card_w
    total_cards_h = cards_per_col * card_h

    available_w = page_w - 2 * margin
    available_h = page_h - 2 * margin

    scale_w = available_w / total_cards_w
    scale_h = available_h / total_cards_h
    scale = min(scale_w, scale_h, 1.0)

    card_w_scaled = card_w * scale
    card_h_scaled = card_h * scale

    grid_w = card_w_scaled * cards_per_row
    grid_h = card_h_scaled * cards_per_col

    start_x = (page_w - grid_w) / 2.0
    start_y = (page_h - grid_h) / 2.0

    def draw_one_page(page_images: List[Path]):
        for index, img_path in enumerate(page_images):
            row = index // cards_per_row
            col = index % cards_per_row

            x = start_x + col * card_w_scaled
            y = start_y + (cards_per_col - 1 - row) * card_h_scaled

            try:
                c.drawImage(
                    str(img_path),
                    x,
                    y,
                    width=card_w_scaled,
                    height=card_h_scaled,
                    preserveAspectRatio=True,
                    anchor='c'
                )
            except Exception as e:
                click.echo(f"⚠️ Erreur lors du dessin de {img_path.name} dans le PDF : {e}", err=True)

        c.showPage()

    per_page = cards_per_row * cards_per_col
    for i in range(0, len(images), per_page):
        page_imgs = images[i:i + per_page]
        draw_one_page(page_imgs)

    c.save()
    click.echo(f"✅ PDF généré : {pdf_path}")


# ---------- Pipeline générique : DeckEntry -> Scryfall -> JSON -> PNG -> PDF ----------

async def entries_to_png_async(
    entries: List[DeckEntry],
    output_dir: Path,
    concurrency: int = 32,
    generate_pdf_flag: bool = True,
    pdf_output_name: str = "cards.pdf",
    host: str = "localhost",
    port: int = 3000,
) -> None:
    click.echo(f"Nombre de cartes à traiter : {len(entries)}")

    if not entries:
        click.echo("Rien à faire, pas de cartes après parsing.", err=True)
        return

    config = Config()
    config.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = output_dir / "render_format"
    render_dir.mkdir(parents=True, exist_ok=True)

    scryfall_cards: List[Optional[dict]] = [None] * len(entries)

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def fetch_index(i: int, entry: DeckEntry):
            async with sem:
                scryfall_cards[i] = await fetch_scryfall_card(session, entry)

        await asyncio.gather(*(fetch_index(i, e) for i, e in enumerate(entries)))

    json_files: List[Path] = []

    for entry, scry in zip(entries, scryfall_cards):
        if scry is None:
            click.echo(
                f"⚠️ Skip {entry.name} ({entry.set_code} {entry.collector_number}) : pas de données Scryfall",
                err=True,
            )
            continue

        render_data = map_scryfall_to_render_json(scry)
        render_data["original_name"] = entry.name

        safe_name = (
            entry.name.replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
        )
        filename = f"{safe_name}_{entry.set_code.upper()}_{entry.collector_number}_render.json"
        json_path = render_dir / filename
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(render_data, f, indent=2)
        json_files.append(json_path)

    click.echo(f"Wrote {len(json_files)} render JSON files to {render_dir}")

    if not json_files:
        click.echo("Aucun JSON de rendu généré, arrêt.", err=True)
        return

    renderer = MTGCardRendererMtgrender(config, host=host, port=port)
    rendered_images = await renderer.render_card_files(json_files)

    click.echo(f"Rendered {len(rendered_images)} PNG files to {output_dir / 'rendered_cards'}")

    if generate_pdf_flag:
        images_dir = output_dir / "rendered_cards"
        pdf_path = output_dir / pdf_output_name
        generate_pdf_from_images(images_dir, pdf_path)


# ---------- CLI (click) ----------

@click.command(
    help=(
        "Rend une liste de cartes MTG en PNG HD en utilisant Scryfall et une instance de mtgrender (Vue.js), "
        "puis génère un PDF prêt à imprimer (3x3, 63x88mm).\n\n"
        "Deux modes d'entrée :\n"
        "  • --text-deck-list --input deck.txt  (decklist Moxfield texte)\n"
        "  • --url-deck-list  --url   URL_Moxfield (deck Moxfield via API v2)\n"
    )
)
@click.option(
    "--text-deck-list",
    "use_text",
    is_flag=True,
    default=False,
    help="Utiliser un fichier decklist texte (format Moxfield) comme entrée.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Chemin vers le fichier decklist texte (à utiliser avec --text-deck-list).",
)
@click.option(
    "--url-deck-list",
    "use_url",
    is_flag=True,
    default=False,
    help="Utiliser une URL Moxfield.com comme entrée (API v2).",
)
@click.option(
    "--url",
    "deck_url",
    type=str,
    help="URL du deck Moxfield (à utiliser avec --url-deck-list).",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/cube_render"),
    show_default=True,
    help="Dossier de sortie pour les JSON de rendu, PNG, et PDF.",
)
@click.option(
    "-c",
    "--concurrency",
    type=int,
    default=32,
    show_default=True,
    help="Nombre maximum de requêtes Scryfall parallèles.",
)
@click.option(
    "--no-pdf",
    "no_pdf",
    is_flag=True,
    default=False,
    help="Ne pas générer de PDF, seulement les PNG.",
)
@click.option(
    "--pdf-output",
    "pdf_output_name",
    type=str,
    default="cards.pdf",
    show_default=True,
    help="Nom du fichier PDF généré (dans output-dir).",
)
@click.option(
    "--host",
    "host",
    type=str,
    default="localhost",
    show_default=True,
    help="Hôte de l'instance mtgrender (par ex. 0.0.0.0, ou un hostname distant).",
)
@click.option(
    "--port",
    "port",
    type=int,
    default=3000,
    show_default=True,
    help="Port de l'instance mtgrender.",
)
def main(
    use_text: bool,
    input_path: Optional[Path],
    use_url: bool,
    deck_url: Optional[str],
    output_dir: Path,
    concurrency: int,
    no_pdf: bool,
    pdf_output_name: str,
    host: str,
    port: int,
) -> None:
    if use_text == use_url:
        raise click.UsageError(
            "Tu dois choisir exactement un mode d'entrée : "
            "--text-deck-list ou --url-deck-list."
        )

    generate_pdf_flag = not no_pdf

    if use_text:
        if not input_path:
            raise click.UsageError(
                "Avec --text-deck-list, tu dois fournir --input PATH vers un fichier texte."
            )
        if not input_path.exists():
            raise click.BadParameter(f"Le fichier decklist n'existe pas: {input_path}")
        entries = parse_text_decklist_file(input_path)
        asyncio.run(entries_to_png_async(
            entries,
            output_dir,
            concurrency=concurrency,
            generate_pdf_flag=generate_pdf_flag,
            pdf_output_name=pdf_output_name,
            host=host,
            port=port,
        ))
        return

    if not deck_url:
        raise click.UsageError(
            "Avec --url-deck-list, tu dois fournir --url URL_Moxfield."
        )

    async def run_url_mode():
        data = await fetch_moxfield_deck_json(deck_url)
        entries = parse_moxfield_json_to_entries(data)
        await entries_to_png_async(
            entries,
            output_dir,
            concurrency=concurrency,
            generate_pdf_flag=generate_pdf_flag,
            pdf_output_name=pdf_output_name,
            host=host,
            port=port,
        )

    asyncio.run(run_url_mode())


if __name__ == "__main__":
    main()
