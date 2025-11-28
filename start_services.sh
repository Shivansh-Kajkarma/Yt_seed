#!/bin/bash
# Quick start script for YTSEED pipeline

echo "🚀 Starting YTSEED Pipeline Services..."
echo ""

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "📦 Starting Redis..."
    redis-server --daemonize yes
    sleep 2
    if pgrep -x "redis-server" > /dev/null; then
        echo "   ✅ Redis started"
    else
        echo "   ❌ Redis failed to start"
        exit 1
    fi
else
    echo "   ✅ Redis already running"
fi

# Activate conda environment
echo ""
echo "🐍 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate video_creator

# Start Celery Worker
echo ""
echo "⚙️  Starting Celery Worker..."
cd /home/rareboy/Internship/Kajkarma/YTSEED
celery -A celery_worker worker --loglevel=info --detach

sleep 2
if pgrep -f "celery.*worker" > /dev/null; then
    echo "   ✅ Celery worker started"
else
    echo "   ❌ Celery worker failed to start"
    exit 1
fi

# Start FastAPI
echo ""
echo "🌐 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &

sleep 3
if curl -s http://localhost:8000/progress > /dev/null; then
    echo "   ✅ FastAPI server started at http://localhost:8000"
else
    echo "   ❌ FastAPI server failed to start"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ ALL SERVICES RUNNING!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "📝 API Endpoints:"
echo "   • Start Pipeline: POST http://localhost:8000/start_pipeline"
echo "   • Check Progress: GET  http://localhost:8000/progress"
echo "   • Get Status:     GET  http://localhost:8000/status/{task_id}"
echo "   • Download:       GET  http://localhost:8000/download_tier1_2"
echo ""
echo "📚 Documentation:"
echo "   • DEPLOYMENT_READY.md  - Complete deployment guide"
echo "   • INTEGRATION_STATUS.md - Integration details"
echo "   • NEW_PIPELINE_FLOW.md  - Pipeline architecture"
echo ""
echo "🧪 Test Command:"
echo "   curl -X POST 'http://localhost:8000/start_pipeline' \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"sheet_url\":\"YOUR_URL\", \"input_format\":\"Podcast\", \"clients_intent\":\"Tech\"}'"
echo ""
echo "🛑 Stop Services:"
echo "   ./stop_services.sh"
echo ""
