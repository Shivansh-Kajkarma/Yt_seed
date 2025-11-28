#!/usr/bin/env python3
"""
Quick integration test for the new YTSEED pipeline
Tests if all components are properly connected
"""

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Imports...")

    try:
        from utils.pipeline_wrapper import full_pipeline_from_sheet

        print("  ✅ pipeline_wrapper imported")
    except ImportError as e:
        print(f"  ❌ pipeline_wrapper import failed: {e}")
        return False

    try:
        from celery_worker import run_phase_pipeline, celery_app

        print("  ✅ celery_worker imported")
    except ImportError as e:
        print(f"  ❌ celery_worker import failed: {e}")
        return False

    try:
        from main import app

        print("  ✅ FastAPI app imported")
    except ImportError as e:
        print(f"  ❌ FastAPI app import failed: {e}")
        return False

    try:
        from utils.fingerprint_llm_utils import gpt_client, get_openai_embedding

        print("  ✅ fingerprint_llm_utils imported")
    except ImportError as e:
        print(f"  ❌ fingerprint_llm_utils import failed: {e}")
        return False

    return True


def test_phase_scripts():
    """Test if all phase scripts exist and are executable"""
    print("\n🧪 Testing Phase Scripts...")

    scripts = [
        "ytdlp_scripts/phase1_seed_processing.py",
        "SCRIPTS/phase2_get_discovered_channels.py",
        "ytdlp_scripts/phase3_step1_api_filter.py",
        "ytdlp_scripts/phase3_step2_deep_scan.py",
        "ytdlp_scripts/phase3_step3a_scoring_llm.py",
        "ytdlp_scripts/phase3_step3b_scoring_emb.py",
        "ytdlp_scripts/phase4_final_ranking.py",
    ]

    all_exist = True
    for script in scripts:
        full_path = BASE_DIR / script
        if full_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} NOT FOUND")
            all_exist = False

    return all_exist


def test_env_vars():
    """Test if required environment variables are set"""
    print("\n🧪 Testing Environment Variables...")

    from dotenv import load_dotenv

    load_dotenv()

    required_vars = [
        "YOUTUBE_API_KEY",
        "OPENAI_API_KEY",
        "MONGO_URI",
        "MONGO_DB_NAME",
        "REDIS_URL",
    ]

    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print actual values for security
            print(f"  ✅ {var} (set)")
        else:
            print(f"  ❌ {var} (not set)")
            all_set = False

    return all_set


def test_connections():
    """Test Redis and MongoDB connections"""
    print("\n🧪 Testing Connections...")

    # Test Redis
    try:
        import redis

        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        print("  ✅ Redis connection")
    except Exception as e:
        print(f"  ❌ Redis connection failed: {e}")
        return False

    # Test MongoDB
    try:
        from pymongo import MongoClient

        client = MongoClient(os.getenv("MONGO_URI"))
        client.server_info()
        print("  ✅ MongoDB connection")
    except Exception as e:
        print(f"  ❌ MongoDB connection failed: {e}")
        return False

    return True


def test_api_keys():
    """Test if API keys are valid (without making actual requests)"""
    print("\n🧪 Testing API Keys...")

    # Test OpenAI
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Just test initialization, not actual API call
        print("  ✅ OpenAI client initialized")
    except Exception as e:
        print(f"  ❌ OpenAI initialization failed: {e}")
        return False

    # Test YouTube API key format
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_key and len(youtube_key) > 10:
        print("  ✅ YouTube API key format valid")
    else:
        print("  ❌ YouTube API key invalid format")
        return False

    return True


def test_pipeline_signature():
    """Test if pipeline functions have correct signatures"""
    print("\n🧪 Testing Function Signatures...")

    try:
        from utils.pipeline_wrapper import (
            full_pipeline_from_sheet,
            run_feedback_loop_for_seed,
        )
        import inspect

        # Check full_pipeline_from_sheet
        sig = inspect.signature(full_pipeline_from_sheet)
        params = list(sig.parameters.keys())
        expected = [
            "celery_task",
            "sheet_url",
            "seed_dict",
            "input_format",
            "clients_intent",
        ]

        if params == expected:
            print(f"  ✅ full_pipeline_from_sheet signature correct")
        else:
            print(f"  ❌ full_pipeline_from_sheet signature mismatch")
            print(f"     Expected: {expected}")
            print(f"     Got: {params}")
            return False

        # Check run_feedback_loop_for_seed
        sig2 = inspect.signature(run_feedback_loop_for_seed)
        params2 = list(sig2.parameters.keys())
        expected2 = ["run_tag", "input_format", "clients_intent"]

        if params2 == expected2:
            print(f"  ✅ run_feedback_loop_for_seed signature correct")
        else:
            print(f"  ❌ run_feedback_loop_for_seed signature mismatch")
            print(f"     Expected: {expected2}")
            print(f"     Got: {params2}")
            return False

        return True
    except Exception as e:
        print(f"  ❌ Signature test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 YTSEED Pipeline Integration Test")
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Phase Scripts": test_phase_scripts(),
        "Environment Variables": test_env_vars(),
        "Connections": test_connections(),
        "API Keys": test_api_keys(),
        "Function Signatures": test_pipeline_signature(),
    }

    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Pipeline is ready!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start Redis: redis-server")
        print("2. Start Celery: celery -A celery_worker worker --loglevel=info")
        print("3. Start FastAPI: uvicorn main:app --reload --port 8000")
        print("4. Test with: curl -X POST 'http://localhost:8000/start_pipeline' \\")
        print("              -H 'Content-Type: application/json' \\")
        print(
            '              -d \'{"sheet_url":"...", "input_format":"Podcast", "clients_intent":"Tech"}\''
        )
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before deploying")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
