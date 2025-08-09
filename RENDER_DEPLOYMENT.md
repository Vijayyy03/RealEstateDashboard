# Deploying the Real Estate Investment System on Render

This guide provides comprehensive instructions for deploying the Real Estate Investment System on Render.com with PostgreSQL database support.

## Prerequisites

1. A [Render](https://render.com) account
2. Git repository with your Real Estate Investment System code
3. Basic familiarity with PostgreSQL and web services

## Files Overview

The following files are configured for Render deployment:

- `render.yaml` - Blueprint configuration for automated deployment
- `Procfile` - Process definitions for web services
- `runtime.txt` - Python version specification
- `requirements.txt` - Python dependencies
- `scripts/init_render_db.py` - Database initialization script
- `scripts/init_render.sh` - Shell script for database setup
- `.gitignore` - Excludes unnecessary files from deployment

## Deployment Options

You have two options for deploying on Render:

1. **Blueprint Deployment** (Recommended): Use the `render.yaml` file for automated deployment
2. **Manual Deployment**: Deploy each service individually through the Render Dashboard

## Option 1: Blueprint Deployment (Recommended)

### Step 1: Prepare Your Repository

1. Ensure all deployment files are in your repository:
   - `render.yaml`
   - `Procfile`
   - `runtime.txt`
   - `requirements.txt`
   - `scripts/init_render_db.py`
   - `scripts/init_render.sh`

2. Commit and push all changes to your Git repository

### Step 2: Deploy via Blueprint

1. Log in to your Render account
2. Navigate to the **New** dropdown and select **Blueprint**
3. Connect your Git repository
4. Select the branch containing your code (usually `main`)
5. Review the services that will be created:
   - PostgreSQL database (`real-estate-db`)
   - API service (`real-estate-api`)
   - Dashboard service (`real-estate-dashboard`)
6. Click **Apply Blueprint**

Render will automatically:
- Create a PostgreSQL database with PostGIS support
- Deploy the API service with Gunicorn and Uvicorn
- Deploy the Streamlit dashboard
- Run the database initialization script

## Option 2: Manual Deployment

### Step 1: Create a PostgreSQL Database

1. Log in to your Render account
2. Navigate to the **New** dropdown and select **PostgreSQL**
3. Configure your database:
   - **Name**: `real-estate-db`
   - **Database**: `real_estate_db`
   - **User**: `real_estate_user`
   - **Plan**: Starter (or higher based on needs)
   - **Region**: Choose the region closest to your users
   - **PostgreSQL Version**: 15
4. Click **Create Database**
5. Once created, note the **Internal Database URL** for later use

### Step 2: Deploy the API Service

1. Navigate to the **New** dropdown and select **Web Service**
2. Connect your Git repository
3. Configure the service:
   - **Name**: `real-estate-api`
   - **Environment**: Python
   - **Region**: Same as your database
   - **Branch**: `main` (or your preferred branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT`
   - **Plan**: Starter (or higher based on needs)
4. Add the following environment variables:
   - `DATABASE_URL`: Paste the Internal Database URL from Step 1
   - `ENVIRONMENT`: `production`
   - `API_HOST`: `0.0.0.0`
   - `API_PORT`: `$PORT`
   - `PYTHON_VERSION`: `3.9.18`
   - `API_DATA_ENABLED`: `true`
5. Click **Create Web Service**

### Step 3: Deploy the Dashboard Service

1. Navigate to the **New** dropdown and select **Web Service**
2. Connect your Git repository (same as API)
3. Configure the service:
   - **Name**: `real-estate-dashboard`
   - **Environment**: Python
   - **Region**: Same as your database
   - **Branch**: `main` (or your preferred branch)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/main.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false`
   - **Plan**: Starter (or higher based on needs)
4. Add the following environment variables:
   - `DATABASE_URL`: Paste the Internal Database URL from Step 1
   - `ENVIRONMENT`: `production`
   - `API_URL`: URL of your API service (e.g., `https://real-estate-api.onrender.com`)
   - `PYTHON_VERSION`: `3.9.18`
5. Click **Create Web Service**

## Post-Deployment Steps

### Initialize the Database

After deployment, initialize the database with tables and sample data:

#### For Blueprint Deployment:
The database initialization runs automatically via the `postDeploy` hook in `render.yaml`.

#### For Manual Deployment:
1. Access your API service shell in the Render dashboard
2. Run the initialization script:
   ```bash
   python scripts/init_render_db.py
   ```

### Verify Deployment

1. **API Health Check**: Visit `https://your-api-service.onrender.com/health`
2. **API Documentation**: Visit `https://your-api-service.onrender.com/docs` for Swagger UI
3. **Dashboard**: Visit `https://your-dashboard-service.onrender.com`
4. **Test Endpoints**: Try a few API endpoints to ensure database connectivity

## Environment Variables

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:port/db` |
| `ENVIRONMENT` | Application environment | `production` |
| `PYTHON_VERSION` | Python runtime version | `3.9.18` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `$PORT` |
| `API_DATA_ENABLED` | Enable data ingestion features | `true` |
| `API_URL` | API service URL (for dashboard) | Auto-detected |
| `DASHBOARD_URL` | Dashboard URL (for API) | Auto-detected |

## Troubleshooting

### Database Connection Issues

**Symptoms**: Application fails to start, database connection errors

**Solutions**:
1. Verify the `DATABASE_URL` environment variable is correctly set
2. Check if the database service is running in the Render dashboard
3. Ensure the database and web services are in the same region
4. Check database logs for connection issues

### Application Startup Errors

**Symptoms**: Service fails to start, build errors

**Solutions**:
1. Check the service logs in the Render dashboard
2. Verify all required environment variables are set
3. Ensure `requirements.txt` includes all dependencies
4. Check Python version compatibility (3.9.x recommended)
5. Verify the start command syntax in `Procfile`

### Dashboard Not Loading

**Symptoms**: Dashboard shows errors, blank page, or connection issues

**Solutions**:
1. Check if the API service is running and accessible
2. Verify the `API_URL` environment variable points to the correct API service
3. Check Streamlit service logs for specific errors
4. Ensure CORS settings are properly configured

### Performance Issues

**Symptoms**: Slow response times, timeouts

**Solutions**:
1. Upgrade to a higher-tier plan for more resources
2. Optimize database queries and add indexes
3. Consider implementing caching strategies
4. Monitor resource usage in the Render dashboard

### Build Failures

**Symptoms**: Deployment fails during build phase

**Solutions**:
1. Check for syntax errors in `requirements.txt`
2. Ensure all dependencies are compatible with Python 3.9
3. Verify file paths and imports are correct
4. Check build logs for specific error messages

## Monitoring and Maintenance

### Monitoring

- **Service Health**: Use the `/health` endpoint for API monitoring
- **Logs**: Monitor service logs in the Render dashboard
- **Metrics**: Track resource usage and performance metrics
- **Alerts**: Set up alerts for service failures or high resource usage

### Maintenance Tasks

- **Dependencies**: Regularly update Python packages in `requirements.txt`
- **Database**: Monitor database performance and storage usage
- **Backups**: Ensure database backups are configured and tested
- **Security**: Keep dependencies updated for security patches

### Scaling

- **Vertical Scaling**: Upgrade to higher-tier plans for more CPU/memory
- **Horizontal Scaling**: Use Render's auto-scaling features (available on higher plans)
- **Database Scaling**: Upgrade database plan as data grows

## Cost Optimization

- **Free Tier**: Use free tier for development/testing
- **Starter Plans**: Suitable for small to medium applications
- **Resource Monitoring**: Monitor usage to avoid unnecessary costs
- **Sleep Mode**: Services on free tier sleep after inactivity

## Security Best Practices

- **Environment Variables**: Never commit secrets to your repository
- **Database Access**: Use strong passwords and limit IP access if needed
- **HTTPS**: All Render services use HTTPS by default
- **Dependencies**: Keep all packages updated for security

---

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [PostgreSQL on Render](https://render.com/docs/databases)
- [Python on Render](https://render.com/docs/python)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Blueprints](https://render.com/docs/blueprint-spec)

For technical support, contact your system administrator or refer to the Render support documentation.