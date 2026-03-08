import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness
)
from src.generation import get_rag_chain

load_dotenv()

def test_offline_rag_evaluation():
   
    chain = get_rag_chain("chroma_db")

    csv_path = "tests/ground_truth.csv"
    df_gt = pd.read_csv(csv_path)

    results = []
    
    print(f"--- Starting evaluation for {len(df_gt)} questions ---")

    for index, row in df_gt.iterrows():
        
        query = row['Question']
        ground_truth = row['Ground Truth (Reference)']
        
        print(f"Evaluating Q{index+1}: {query[:50]}...")
        
       
        response = chain.invoke({"query": query})
        
        
        results.append({
            "user_input": query,             # New standard for 'question'
            "response": response["result"],  # New standard for 'answer'
            "retrieved_contexts": [doc.page_content for doc in response["source_documents"]], # For 'contexts'
            "reference": ground_truth        # New standard for 'ground_truth'
        })

    
    eval_dataset = Dataset.from_pandas(pd.DataFrame(results))

    
    score = evaluate(
        eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness()
        ]
    )

   
    print("\n" + "="*30)
    print("FINAL RAGAS SCORES")
    print("="*30)
    print(score)
    
    
    df_scores = score.to_pandas()
    df_scores.to_csv("evaluation_results_detailed.csv", index=False)
    print("\nDetailed results saved to 'evaluation_results_detailed.csv'")

    
