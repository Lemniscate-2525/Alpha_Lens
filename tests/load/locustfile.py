"""
AlphaLens - Locust Load Testing Suite
Simulates realistic 10,000-user traffic profile with peak earnings release spikes.
"""
import random
import json
import random
from locust import FastHttpUser, task, between


SAMPLE_QUERIES = [
    {"query": "What was NVIDIA's revenue and gross margin in Q2?", "ticker": "NVDA"},
    {"query": "What did CFO Colette Kress say about Blackwell gross margin dilution in Q4?", "ticker": "NVDA"},
    {"query": "What was Data Center revenue growth year over year in Q3?", "ticker": "NVDA"},
    {"query": "What was Microsoft Cloud revenue and Azure growth in Q4?", "ticker": "MSFT"},
    {"query": "What did Jensen Huang explain regarding the Blackwell mask change?", "ticker": "NVDA"},
    {"query": "What did Amy Hood identify as the primary capacity constraint in Azure?", "ticker": "MSFT"},
    {"query": "What was GAAP diluted EPS in the Q3 10-Q filing?", "ticker": "NVDA"},
    {"query": "What was NVIDIA's cash and cash equivalents balance?", "ticker": "NVDA"},
]


class FinancialAnalystUser(FastHttpUser):
    wait_time = between(0.1, 0.4)
    default_headers = {
        "X-API-Key": "alpha-lens-master-key-prod-2025",
        "Content-Type": "application/json"
    }

    @task(8)
    def query_rag_pipeline(self):
        sample = random.choice(SAMPLE_QUERIES)
        payload = {
            "query": sample["query"],
            "ticker": sample["ticker"],
            "top_k": 3,
            "use_reranker": False,
            "use_semantic_cache": True
        }
        self.client.post(
            "/api/v1/query",
            data=json.dumps(payload),
            headers=self.default_headers,
            name="/api/v1/query"
        )

    @task(1)
    def check_health(self):
        self.client.get("/api/v1/health", headers=self.default_headers, name="/api/v1/health")

    @task(1)
    def check_metrics(self):
        self.client.get("/api/v1/metrics", headers=self.default_headers, name="/api/v1/metrics")
