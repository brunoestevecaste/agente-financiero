# Agente Financiero con Streamlit

Aplicacion de analisis financiero que:
- consulta el precio de un activo en la ultima semana,
- extrae noticias del ultimo mes desde Yahoo Finance,
- genera un analisis con posible sesgo de subida/bajada.

## Estructura del proyecto

- `app.py`: aplicacion principal Streamlit.
- `requirements.txt`: dependencias para despliegue.
- `runtime.txt`: version de Python para Streamlit Cloud.
- `.streamlit/config.toml`: configuracion visual y de servidor.
- `.streamlit/secrets.toml.example`: plantilla de secretos.

## Ejecucion local

1. Crear entorno virtual e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configurar clave de Google:

```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

3. Editar `.streamlit/secrets.toml` y poner tu clave real:

```toml
GOOGLE_API_KEY = "tu_clave_real"
```

4. Lanzar la app:

```bash
streamlit run app.py
```

## Crear repositorio y subir a GitHub

Ejemplo con repositorio nuevo:

```bash
git init
git add .
git commit -m "Initial commit: Streamlit financial agent"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

## Despliegue en Streamlit Community Cloud

1. Entra en https://share.streamlit.io y conecta tu cuenta de GitHub.
2. Pulsa **New app**.
3. Selecciona:
- Repository: `TU_USUARIO/TU_REPO`
- Branch: `main`
- Main file path: `app.py`
4. En **Advanced settings > Secrets**, pega:

```toml
GOOGLE_API_KEY = "tu_clave_real"
```

5. Deploy.

Si no defines el secreto, la app te pedira la API key manualmente en el sidebar.
