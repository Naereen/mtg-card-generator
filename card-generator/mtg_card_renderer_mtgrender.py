import json
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlencode

from playwright.async_api import async_playwright

from models import Config


class MTGCardRendererMtgrender:
    """
    Renderer utilisant une instance de mtgrender (Vue.js), par défaut sur
    http://localhost:3000/mtgrender/

    Pour chaque carte, on construit une URL du style :

        http://{host}:{port}/mtgrender/?card=Hostage%20Taker&set=mb2&cn=137

    puis on screenshot le conteneur .card-display.
    """

    CARD_SELECTOR = ".card-display"

    def __init__(
        self,
        config: Config,
        host: str = "localhost",
        port: int = 3000,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}/mtgrender"

    async def render_card_files(self, json_files: Iterable[Path]) -> List[Path]:
        json_files = list(json_files)
        if not json_files:
            print("No card files to render (mtgrender)")
            return []

        rendered_images: List[Path] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(
                viewport={"width": 1024, "height": 1500},
                device_scale_factor=2,  # HD
            )

            for json_file in json_files:
                with json_file.open("r", encoding="utf-8") as f:
                    card_data = json.load(f)

                name: Optional[str] = card_data.get("name")
                set_code: Optional[str] = card_data.get("set")
                collector_number: Optional[str] = card_data.get("collector_number")

                if not name or not set_code or not collector_number:
                    print(
                        f"⚠️ {json_file.name}: champs manquants pour construire l'URL mtgrender "
                        f"(name={name}, set={set_code}, cn={collector_number}), skip."
                    )
                    continue

                query = urlencode(
                    {
                        "card": name,
                        "set": set_code,
                        "cn": collector_number,
                    }
                )
                url = f"{self.base_url}/?{query}"
                print(f"→ Rendu via mtgrender: {url}")

                # Charger la page pour cette carte
                await page.goto(url)
                await page.wait_for_load_state("networkidle")

                # Attendre que la carte soit affichée dans .card-display
                try:
                    await page.wait_for_selector(self.CARD_SELECTOR, state="visible", timeout=15000)
                except Exception as e:
                    print(f"⚠️ .card-display non visible pour {name} ({set_code} {collector_number}) : {e}")
                    continue

                # Laisser un peu de temps pour que les fonts / images se stabilisent
                await page.wait_for_timeout(300)

                card_element = await page.query_selector(self.CARD_SELECTOR)
                if not card_element:
                    print(f"⚠️ Impossible de trouver {self.CARD_SELECTOR} pour {name}")
                    continue

                box = await card_element.bounding_box()
                if not box:
                    print(f"⚠️ Pas de bounding box pour {name}")
                    continue

                rendered_cards_dir = self.config.output_dir / "rendered_cards"
                rendered_cards_dir.mkdir(parents=True, exist_ok=True)

                base_name = card_data.get("original_name") or card_data.get("name") or json_file.stem
                safe_name = (
                    base_name.replace(" ", "_")
                    .replace("/", "_")
                    .replace(":", "_")
                )
                output_path = rendered_cards_dir / f"{safe_name}.png"

                await page.screenshot(
                    path=str(output_path),
                    clip=box,
                    type="png",
                    omit_background=False,
                    animations="disabled",
                )

                rendered_images.append(output_path)

            await browser.close()

        return rendered_images
