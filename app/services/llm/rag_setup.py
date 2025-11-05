# app/services/llm/rag_setup.py
"""
RAG Setup Script - Builds vector store from your SQLite KB and YAML rules
Run this ONCE to create the vector store, then use it in the agent
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import sqlite3

class KnowledgeBaseRAG:
    def __init__(self, db_path: str = "supply_chain.db", vector_store_path: str = "vector_store"):
        self.db_path = db_path
        self.vector_store_path = vector_store_path
        self.documents = []
        
    def load_model_metadata(self) -> List[Document]:
        """Load model metadata from SQLite database"""
        print("📊 Loading model metadata from SQLite...")
        documents = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all models with their metadata
            cursor.execute("""
                SELECT model_name, model_type, model_path, required_features, 
                    optional_features, target_variable, performance_metrics,
                    hyperparameters, description
                FROM ML_Models  # Your actual table name
                WHERE is_active = TRUE
            """)
                        
            models = cursor.fetchall()
            
            for model in models:
                model_name, model_type, description, req_features, opt_features, \
                target_var, perf_metrics, use_cases, limitations = model
                
                # Create rich document for each model
                content = f"""
Model: {model_name}
Type: {model_type}
Description: {description}

Required Features: {req_features}
Optional Features: {opt_features}
Target Variable: {target_var}

Use Cases: {use_cases}
Limitations: {limitations}

Performance Metrics: {perf_metrics}
"""
                
                metadata = {
                    "source": "model_metadata",
                    "model_name": model_name,
                    "model_type": model_type,
                    "required_features": req_features,
                    "target_variable": target_var
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            conn.close()
            print(f"✅ Loaded {len(documents)} model documents")
            
        except Exception as e:
            print(f"❌ Error loading model metadata: {e}")
        
        return documents
    


    def load_yaml_rules(self) -> List[Document]:
        """Load YAML rules for validation and model selection"""
        print("📋 Loading YAML rules...")
        documents = []
        
        yaml_files = [
            "app/knowledge_base/rule_layer/model_selection_rules.yaml",
            "app/knowledge_base/rule_layer/data_validation_rules.yaml"
        ]
        
        for yaml_file in yaml_files:
            try:
                if not Path(yaml_file).exists():
                    print(f"⚠️ File not found: {yaml_file}")
                    continue
                
                with open(yaml_file, 'r') as f:
                    rules = yaml.safe_load(f)
                
                # Convert rules to readable text
                rule_type = "model_selection" if "model_selection" in yaml_file else "data_validation"
                content = f"Rule Type: {rule_type}\n\n"
                content += yaml.dump(rules, default_flow_style=False)
                
                metadata = {
                    "source": "yaml_rules",
                    "rule_type": rule_type,
                    "file": yaml_file
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                print(f"✅ Loaded rules from {yaml_file}")
                
            except Exception as e:
                print(f"❌ Error loading {yaml_file}: {e}")
        
        return documents
    
    def load_forecasting_knowledge(self) -> List[Document]:
        """Load general forecasting domain knowledge"""
        print("📚 Loading forecasting domain knowledge...")
        
        knowledge_base = [
            {
                "title": "Time Series Forecasting Basics",
                "content": """
Time series forecasting predicts future values based on historical patterns.
Key components:
- Trend: Long-term increase or decrease
- Seasonality: Repeating patterns at fixed intervals
- Cyclicality: Non-fixed repeating patterns
- Irregularity: Random noise

Common requirements:
- Minimum 24 data points for monthly data
- Regular time intervals (no gaps)
- Numeric target variable
- Date/time column
"""
            },
            {
                "title": "Model Selection Guidelines",
                "content": """
Choosing the right forecasting model:

ARIMA: Best for univariate time series with clear trends/seasonality
- Requires: Regular intervals, minimum 50 observations
- Use when: Simple patterns, no external variables

Prophet: Good for daily data with strong seasonality
- Requires: Date column, minimum 2+ seasonality cycles
- Use when: Multiple seasonality, holidays matter

LightGBM: Best for complex patterns with many features
- Requires: Multiple features, sufficient data (100+ rows)
- Use when: Non-linear patterns, feature engineering available

TFT (Temporal Fusion Transformer): Advanced neural network
- Requires: Large datasets (1000+ observations), multiple features
- Use when: Complex temporal dependencies, attention needed
"""
            },
            {
                "title": "Data Quality Requirements",
                "content": """
Essential data quality checks:

1. Missing Values: Maximum 10% missing data acceptable
2. Outliers: Detect and handle extreme values
3. Data Types: Date columns as datetime, target as numeric
4. Completeness: No gaps in time series
5. Consistency: Same frequency throughout (daily, weekly, monthly)

Common issues and fixes:
- Missing dates: Interpolate or forward-fill
- Outliers: Cap at percentiles or use robust methods
- Mixed frequencies: Resample to consistent interval
"""
            },
            {
                "title": "Forecast Horizon Guidelines",
                "content": """
Recommended forecast horizons by data frequency:

Daily data: Up to 30-90 days ahead
Weekly data: Up to 12-24 weeks ahead
Monthly data: Up to 12-24 months ahead

Factors affecting horizon:
- Model complexity (simpler = longer horizon possible)
- Data quality (better quality = longer horizon)
- Business requirements (demand planning vs tactical decisions)
- Uncertainty tolerance (shorter = more accurate)
"""
            }
        ]
        
        documents = []
        for item in knowledge_base:
            metadata = {
                "source": "domain_knowledge",
                "title": item["title"]
            }
            documents.append(Document(page_content=item["content"], metadata=metadata))
        
        print(f"✅ Loaded {len(documents)} knowledge documents")
        return documents
    
    def build_vector_store(self):
        """Build complete vector store from all sources"""
        print("🔨 Building RAG vector store...\n")
        
        # Collect all documents
        all_documents = []
        all_documents.extend(self.load_model_metadata())
        all_documents.extend(self.load_yaml_rules())
        all_documents.extend(self.load_forecasting_knowledge())
        
        if not all_documents:
            raise ValueError("No documents found to build vector store!")
        
        print(f"\n📦 Total documents collected: {len(all_documents)}")
        
        # Create embeddings
        print("🔄 Creating embeddings (this may take a minute)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Build FAISS vector store
        print("💾 Building FAISS vector store...")
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        
        # Save to disk
        vectorstore.save_local(self.vector_store_path)
        print(f"✅ Vector store saved to {self.vector_store_path}")
        
        return vectorstore
    
    def test_retrieval(self, vectorstore, query: str = "What models can forecast sales?"):
        """Test the retrieval system"""
        print(f"\n🧪 Testing retrieval with query: '{query}'")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        
        print(f"📄 Retrieved {len(docs)} relevant documents:\n")
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
    
    def load_existing_store(self):
        """Load existing vector store from disk"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.load_local(
            self.vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )


def main():
    """Main execution"""
    print("=" * 60)
    print("🚀 RAG Knowledge Base Setup")
    print("=" * 60 + "\n")
    
    # Initialize
    rag = KnowledgeBaseRAG()
    
    # Check if vector store already exists
    if Path(rag.vector_store_path).exists():
        print("⚠️  Vector store already exists!")
        response = input("Rebuild it? (y/n): ")
        if response.lower() != 'y':
            print("Loading existing vector store...")
            vectorstore = rag.load_existing_store()
            rag.test_retrieval(vectorstore)
            return
    
    # Build vector store
    try:
        vectorstore = rag.build_vector_store()
        
        # Test it
        rag.test_retrieval(vectorstore)
        
        print("\n" + "=" * 60)
        print("✅ RAG Setup Complete!")
        print("=" * 60)
        print(f"\nVector store saved to: {rag.vector_store_path}")
        print("You can now use this in your forecast agent.\n")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()