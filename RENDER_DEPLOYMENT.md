# Deploying the Real Estate Investment System on Render

This guide provides instructions for deploying the Real Estate Investment System on Render.com.

## Prerequisites

1. A [Render account](https://render.com/)
2. Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Options

There are two ways to deploy this application on Render:

1. **Manual Deployment**: Deploy each service individually through the Render Dashboard
2. **Blueprint Deployment**: Use the `render.yaml` file for automatic deployment

## Option 1: Manual Deployment

### Step 1: Deploy the PostgreSQL Database

1. Log in to your Render account
2. Navigate to the Dashboard and click on "New" > "PostgreSQL"
3. Configure your database:
   - Name: `real-estate-db`
   - Database: `real_estate_db`
   - User: `real_estate_user`
   - Region: Choose the region closest to your users
   - Plan: Select an appropriate plan (Free tier is available)
4. Click "Create Database"
5. Once created, note the "Internal Database URL" for the next steps

### Step 2: Deploy the API Service

1. In the Render Dashboard, click on "New" > "Web Service"
2. Connect your Git repository
3. Configure the service:
   - Name: `real-estate-api`
   - Environment: `Python`
   - Region: Same as your database
   - Branch: `main` (or your preferred branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add the following environment variables:
   - `ENVIRONMENT`: `production`
   - `DATABASE_URL`: Paste the Internal Database URL from Step 1
   - `API_DATA_ENABLED`: `true`
5. Click "Create Web Service"

### Step 3: Deploy the Dashboard Service

1. In the Render Dashboard, click on "New" > "Web Service"
2. Connect your Git repository (same as before)
3. Configure the service:
   - Name: `real-estate-dashboard`
   - Environment: `Python`
   - Region: Same as your database
   - Branch: `main` (or your preferred branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run dashboard/main.py --server.port $PORT --server.address 0.0.0.0`
4. Add the same environment variables as the API service
5. Click "Create Web Service"

## Option 2: Blueprint Deployment

1. Ensure your repository contains the `render.yaml` file
2. In the Render Dashboard, click on "New" > "Blueprint"
3. Connect your Git repository
4. Render will automatically detect the `render.yaml` file and create all services defined in it
5. Review the configuration and click "Apply"

## Post-Deployment Steps

### Initialize the Database

1. Once all services are deployed, go to the `real-estate-api` service in the Render Dashboard
2. Navigate to the "Shell" tab
3. Run the following command to initialize the database:
   ```
   python database/setup_db.py
   ```
4. Generate sample data (optional):
   ```
   python scripts/sample_data.py
   ```

### Verify Deployment

1. Access the API at `https://real-estate-api.onrender.com/docs`
2. Access the Dashboard at `https://real-estate-dashboard.onrender.com`

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

1. Verify the `DATABASE_URL` environment variable is correctly set
2. Check that the database service is running
3. Ensure the database URL format is correct (should start with `postgresql://`)

### Application Errors

1. Check the service logs in the Render Dashboard
2. Verify all required environment variables are set
3. Ensure the database is properly initialized

### Streamlit Dashboard Issues

If the Streamlit dashboard doesn't load:

1. Check that the `STREAMLIT_SERVER_ADDRESS` is set to `0.0.0.0`
2. Verify the `PORT` environment variable is being used correctly
3. Check the service logs for any Streamlit-specific errors

## Scaling and Monitoring

- Render provides automatic scaling for paid plans
- Monitor your application's performance in the Render Dashboard
- Set up alerts for service health and performance

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/deploy/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)