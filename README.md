# WiFi Router Placement Optimizer

A production-grade web application that takes architectural floor plans as input and outputs optimal WiFi router placements to maximize coverage using physics-based RF propagation simulation.

## Features

- **Floor Plan Upload**: Upload PNG, JPG, or PDF floor plans
- **Wall Detection**: Automatic wall detection with manual editing
- **Router Database**: 25+ popular WiFi router models with real specifications
- **Physics-Based Simulation**: RF propagation using Free Space Path Loss + wall attenuation
- **Genetic Algorithm Optimization**: Find optimal router placements
- **Coverage Heatmaps**: Visual signal strength visualization
- **Multiple Solutions**: Pareto-optimal solutions balancing coverage and cost

## Tech Stack

### Backend
- Python 3.11+
- FastAPI (async API framework)
- PostgreSQL 15 with PostGIS
- Redis (caching + Celery broker)
- Celery (background task processing)
- OpenCV (image processing)
- NumPy/SciPy (numerical computation)

### Frontend
- React 18 + TypeScript
- Vite (build tool)
- Zustand (state management)
- React Query (API calls)
- Konva.js (canvas manipulation)
- Tailwind CSS + shadcn/ui

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
cd telcospace

# Start all services
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f api

# The application will be available at:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Manual Setup

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start PostgreSQL and Redis (using Docker)
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_USER=router_user \
  -e POSTGRES_PASSWORD=router_password \
  -e POSTGRES_DB=router_optimizer \
  postgis/postgis:15-3.3

docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run database migrations
alembic upgrade head

# Seed router database
python scripts/seed_routers.py

# Start the API server
uvicorn app.main:app --reload

# In another terminal, start Celery worker
celery -A app.tasks.celery_app worker --loglevel=info
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

## Usage

1. **Upload Floor Plan**: Go to the homepage and upload your floor plan image
2. **Edit Walls**: Use the editor to add/modify walls and set materials
3. **Configure Optimization**: Set constraints (budget, coverage, signal strength)
4. **Select Routers**: Choose specific routers or use all available
5. **Run Optimization**: Click "Start Optimization" and wait for results
6. **View Results**: Browse solutions with coverage heatmaps

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `GET /api/v1/routers/` - List available routers
- `POST /api/v1/projects/` - Create new project with floor plan
- `PUT /api/v1/projects/{id}/map` - Update map data
- `POST /api/v1/optimization/jobs` - Start optimization job
- `GET /api/v1/optimization/jobs/{id}/solutions` - Get optimization results

## Project Structure

```
telcospace/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── core/            # Configuration
│   │   ├── crud/            # Database operations
│   │   ├── db/              # Database setup
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   └── tasks/           # Celery tasks
│   ├── alembic/             # Database migrations
│   ├── scripts/             # Utility scripts
│   └── tests/               # Backend tests
├── frontend/
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── store/           # Zustand store
│   │   └── types/           # TypeScript types
│   └── public/
├── static/
│   ├── uploads/             # Uploaded floor plans
│   └── heatmaps/            # Generated heatmaps
└── docker-compose.yml
```

## Configuration

### Optimization Parameters

- `max_routers`: Maximum number of routers (1-10)
- `max_budget`: Maximum total cost in USD
- `min_coverage_percent`: Minimum coverage percentage (50-100%)
- `min_signal_strength_dbm`: Minimum signal strength (-90 to -50 dBm)
- `prioritize_cost`: Weight cost over coverage

### RF Propagation

The application uses a simplified ray-tracing model with:
- Free Space Path Loss (Friis equation)
- Wall penetration losses (material-specific attenuation)
- Log-distance path loss model (indoor environment)

Wall materials and their attenuation:
- Concrete: 15 dB
- Brick: 12 dB
- Wood: 6 dB
- Glass: 5 dB
- Drywall: 3 dB
- Metal: 25 dB

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Adding New Routers

Edit `backend/scripts/seed_routers.py` and add new router specifications:

```python
{
    "model_name": "Your Router Model",
    "manufacturer": "Manufacturer",
    "frequency_bands": ["2.4GHz", "5GHz"],
    "max_tx_power_dbm": 23,
    "antenna_gain_dbi": 6.0,
    "wifi_standard": "WiFi 6",
    "max_range_meters": 70,
    "coverage_area_sqm": 250,
    "price_usd": 199.99,
    "specs": {"key": "value"}
}
```

## Future Enhancements

- [ ] PyLayers integration for full ray-tracing
- [ ] Multi-floor building support
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] ML-based coverage refinement
- [ ] Export to PDF reports

## License

MIT License
