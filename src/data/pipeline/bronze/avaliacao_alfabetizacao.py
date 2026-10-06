import truststore
import requests
from pathlib import Path
from zipfile import ZipFile

truststore.inject_into_ssl()

BASE_URL = "https://download.inep.gov.br/dados_abertos/"

FILES = {
    "avaliacao_alfabetizacao_2025": "https://download.inep.gov.br/dados_abertos/microdados_AEEB_2025.zip",
    "avaliacao_alfabetizacao_2024": "https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_2024.zip",
    "avaliacao_alfabetizacao_2023": "https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_2023.zip"
}

CHUNK_SIZE = 1024 * 1024  # 1 MB


if __name__ == "__main__":
    for name, url in FILES.items():
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            print(f"Baixando {name} de {url}...")

            total_size = int(response.headers.get('Content-Length', 0))
            downloaded_size = 0

            with open(f"{name}.zip", "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        downloaded_size += len(chunk)
                        percent_complete = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        print(f"Progresso: {percent_complete:.2f}% ({downloaded_size}/{total_size} bytes)", end="\r")
                    f.write(chunk)

                print(f"Download {name} concluído com sucesso!")

        arquivo_zip = Path(f"{name}.zip")
        diretorio_saida = Path(f"data/bronze/{name}")

        with ZipFile(arquivo_zip) as zip_file:
            zip_file.extractall(diretorio_saida)
