1. copy .env.example -> .env and edit if necessary.
2. python -m venv .venv
3. source .venv/bin/activate   (Windows: .venv\Scripts\activate)
4. pip install -r requirements.txt
5. Ensure PostgreSQL is running and accessible by DATABASE_URL in .env
6. Initialize Alembic (not necessary if provided files exist). To create DB tables:
   alembic upgrade head
7. Start FastAPI:
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
8. Start Streamlit:
   streamlit run frontend/streamlit_app.py
