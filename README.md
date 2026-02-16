# Agente Financiero (Streamlit)

App de analisis financiero con IA que:
- obtiene precios diarios de la ultima semana,
- recoge noticias del ultimo mes,
- genera una conclusion de corto plazo (subida, bajada o lateral).

## Requisitos
- Python 3.11
- Google API Key (la introduce el usuario en la propia app)

## Ejecutar en local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Subir a GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

## Despliegue en Streamlit Community Cloud
1. Entra en https://share.streamlit.io
2. Crea una app nueva y selecciona tu repositorio.
3. Configura `Main file path` como `app.py`.
4. Pulsa `Deploy`.

No necesitas `secrets.toml` porque la clave se pide en el sidebar.
