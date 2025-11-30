"""
Main Pipeline Orchestrator for Adaptive Meta-Learning Deepfake Detection

This script orchestrates the complete pipeline:
1. RAG Module: Vector DB creation and maintenance
2. Multi-Agent Workflow: Prompt generation
3. Sample Synthesis: Few-shot sample generation
4. Meta-Learning: Model training with few-shot samples
"""

import os
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Any
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

# Import modules
from RAG_workflow import load_pdfs, get_or_create_vector_db, get_ensemble_retriever
from web_crawling import DFScanner
from multiagents import run_multiagent_workflow
from sample_synthesis_module import run_sample_synthesis_pipeline
import importlib.util
import sys

# Import meta_learning module (handles the + in filename)
meta_learning_path = "meta_learning+adv+aug_sample_generation.py"
spec = importlib.util.spec_from_file_location("meta_learning", meta_learning_path)
meta_learning = importlib.util.module_from_spec(spec)
sys.modules["meta_learning"] = meta_learning
spec.loader.exec_module(meta_learning)
run_meta_learning_pipeline = meta_learning.run_meta_learning_pipeline


class DeepfakeDetectionPipeline:
    """Main pipeline orchestrator for the deepfake detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary with all paths and parameters
        """
        self.config = config
        self.results = {}
        
    def run_rag_module(self, pdf_folder: Optional[str] = None, 
                      run_web_crawl: bool = True):
        """
        Run the RAG module to create and maintain vector database.
        
        Args:
            pdf_folder: Path to folder containing PDFs (optional)
            run_web_crawl: Whether to run web crawling
        """
        print("\n" + "="*60)
        print("MODULE A: RAG - Vector Database Creation")
        print("="*60)
        
        vector_db_path = self.config.get('vector_db_path', './chroma_db')
        
        # Create/load vector DB from PDFs
        if pdf_folder:
            print(f"\nStep 1: Loading PDFs from {pdf_folder}")
            db, embeddings = get_or_create_vector_db(
                persist_directory=vector_db_path,
                pdf_folder=pdf_folder
            )
            print("Vector DB initialized from PDFs")
        else:
            print(f"\nStep 1: Loading existing vector DB from {vector_db_path}")
            db, embeddings = get_or_create_vector_db(
                persist_directory=vector_db_path,
                pdf_folder=None
            )
            print("Vector DB loaded")
        
        # Run web crawling if requested
        if run_web_crawl:
            print("\nStep 2: Running web crawling to fetch latest information")
            firecrawl_key = os.environ.get('FIRECRAWL_API_KEY')
            if not firecrawl_key:
                print("Warning: FIRECRAWL_API_KEY not set. Skipping web crawling.")
            else:
                scanner = DFScanner(
                    api_key=firecrawl_key,
                    vector_db_path=vector_db_path,
                    add_to_vector_db=True
                )
                deep_crawl = self.config.get('deep_crawl', False)
                scanner.run_scan(deep_crawl=deep_crawl)
                db, embeddings = get_or_create_vector_db(
                persist_directory=vector_db_path,
                pdf_folder=scanner.report()
            )
                print("Web crawling completed and content added to vector DB")
        
        self.results['rag'] = {
            'vector_db_path': vector_db_path,
            'status': 'completed'
        }
        print("\nRAG Module completed")
        
    def run_multiagent_module(self, user_query: str):
        """
        Run the multi-agent workflow to generate prompts.
        
        Args:
            user_query: Query for generating deepfake prompts
        
        Returns:
            Dictionary with generated prompts
        """
        print("\n" + "="*60)
        print("MODULE B: Multi-Agent Hierarchical Workflow")
        print("="*60)
        
        output_dir = self.config.get('multiagent_output_dir', './outputs')
        
        print(f"\nUser Query: {user_query}")
        print("Running multi-agent workflow...")
        
        prompts = run_multiagent_workflow(user_query, output_dir)
        
        # If prompts are empty, try to load from file
        if not prompts.get('positive_prompts') and not prompts.get('negative_prompts'):
            from sample_synthesis_module import load_prompts_from_multiagents
            try:
                prompts = load_prompts_from_multiagents(output_dir)
            except Exception as e:
                print(f"Warning: Could not load prompts from file: {e}")
        
        self.results['multiagent'] = {
            'user_query': user_query,
            'output_dir': output_dir,
            'num_positive_prompts': len(prompts.get('positive_prompts', [])),
            'num_negative_prompts': len(prompts.get('negative_prompts', [])),
            'status': 'completed'
        }
        
        print(f"\nMulti-Agent Module completed")
        print(f"  Generated {len(prompts.get('positive_prompts', []))} positive prompts")
        print(f"  Generated {len(prompts.get('negative_prompts', []))} negative prompts")
        
        return prompts
    
    def run_sample_synthesis_module(self, user_query: str, base_image_dir: str):
        """
        Run sample synthesis to generate few-shot samples.
        
        Args:
            user_query: Query for multi-agent workflow
            base_image_dir: Directory containing base images
        
        Returns:
            List of paths to generated samples
        """
        print("\n" + "="*60)
        print("MODULE C: Sample Synthesis")
        print("="*60)
        
        output_dir = self.config.get('few_shot_samples_dir', './few_shot_samples')
        multiagent_output_dir = self.config.get('multiagent_output_dir', './outputs')
        text_prompt = self.config.get('text_prompt', 'person face')
        mask_type = self.config.get('mask_type', 'normal')
        max_samples = self.config.get('max_samples', None)
        
        print(f"\nGenerating few-shot samples...")
        print(f"Base images: {base_image_dir}")
        print(f"Output directory: {output_dir}")
        
        samples = run_sample_synthesis_pipeline(
            user_query=user_query,
            base_image_dir=base_image_dir,
            output_dir=output_dir,
            multiagent_output_dir=multiagent_output_dir,
            text_prompt=text_prompt,
            mask_type=mask_type,
            max_samples=max_samples
        )
        
        self.results['sample_synthesis'] = {
            'output_dir': output_dir,
            'num_samples': len(samples),
            'status': 'completed'
        }
        
        print(f"\nSample Synthesis Module completed!")
        print(f"  Generated {len(samples)} few-shot samples")
        
        return samples
    
    def run_meta_learning_module(self, few_shot_samples_path: Optional[str] = None):
        """
        Run meta-learning training with few-shot samples.
        
        Args:
            few_shot_samples_path: Path to few-shot samples (optional)
        
        Returns:
            Trained model and performance metrics
        """
        print("\n" + "="*60)
        print("MODULE D: Meta-Learning Training")
        print("="*60)
        
        train_real_path = self.config.get('train_real_path')
        train_fake_path = self.config.get('train_fake_path')
        val_real_path = self.config.get('val_real_path')
        val_fake_path = self.config.get('val_fake_path')
        num_meta_epochs = self.config.get('num_meta_epochs', 30)
        tasks_per_epoch = self.config.get('tasks_per_epoch', 10)
        
        # Use few-shot samples path from config if not provided
        if few_shot_samples_path is None:
            few_shot_samples_path = self.config.get('few_shot_samples_dir', './few_shot_samples')
        
        print(f"\nTraining meta-learning model...")
        print(f"Few-shot samples: {few_shot_samples_path}")
        print(f"Meta-epochs: {num_meta_epochs}, Tasks per epoch: {tasks_per_epoch}")
        
        model, metrics = run_meta_learning_pipeline(
            train_real_path=train_real_path,
            train_fake_path=train_fake_path,
            val_real_path=val_real_path,
            val_fake_path=val_fake_path,
            few_shot_samples_path=few_shot_samples_path,
            num_meta_epochs=num_meta_epochs,
            tasks_per_epoch=tasks_per_epoch
        )
        
        # Save model
        model_save_path = self.config.get('model_save_path', './trained_model.pth')
        import torch
        torch.save(model.state_dict(), model_save_path)
        
        self.results['meta_learning'] = {
            'model_path': model_save_path,
            'metrics': {
                'avg_train_loss': sum(metrics['epoch_avg_train_loss']) / len(metrics['epoch_avg_train_loss']),
                'avg_train_acc': sum(metrics['epoch_avg_train_acc']) / len(metrics['epoch_avg_train_acc']),
                'avg_val_loss': sum(metrics['epoch_avg_val_loss']) / len(metrics['epoch_avg_val_loss']),
                'avg_val_acc': sum(metrics['epoch_avg_val_acc']) / len(metrics['epoch_avg_val_acc'])
            },
            'status': 'completed'
        }
        
        print(f"\nMeta-Learning Module completed")
        print(f"  Model saved to {model_save_path}")
        
        return model, metrics
    
    def run_full_pipeline(self, user_query: str, base_image_dir: str,
                         pdf_folder: Optional[str] = None,
                         run_web_crawl: bool = True):
        """
        Run the complete pipeline end-to-end.
        
        Args:
            user_query: Query for generating deepfake prompts
            base_image_dir: Directory containing base images
            pdf_folder: Path to folder containing PDFs (optional)
            run_web_crawl: Whether to run web crawling
        """
        print("\n" + "="*80)
        print("ADAPTIVE META-LEARNING FOR ROBUST DEEPFAKE DETECTION")
        print("Complete Pipeline Execution")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # Module A: RAG
            self.run_rag_module(pdf_folder=pdf_folder, run_web_crawl=run_web_crawl)
            
            # Module B: Multi-Agent Workflow
            prompts = self.run_multiagent_module(user_query)
            
            # Module C: Sample Synthesis
            samples = self.run_sample_synthesis_module(user_query, base_image_dir)
            
            # Module D: Meta-Learning
            model, metrics = self.run_meta_learning_module()
            
            # Save results
            results_path = self.config.get('results_path', './pipeline_results.json')
            self.results['pipeline'] = {
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETE!")
            print("="*80)
            print(f"Results saved to: {results_path}")
            print(f"Total execution time: {datetime.now() - start_time}")
            
        except Exception as e:
            print(f"\nPipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive Meta-Learning Deepfake Detection Pipeline"
    )
    
    # Pipeline configuration
    parser.add_argument('--config', type=str, default='./config.json',
                       help='Path to configuration JSON file')
    parser.add_argument('--user_query', type=str, required=True,
                       help='Query for generating deepfake prompts')
    parser.add_argument('--base_image_dir', type=str, required=True,
                       help='Directory containing base images')
    parser.add_argument('--pdf_folder', type=str, default=None,
                       help='Path to folder containing PDFs (optional)')
    parser.add_argument('--run_web_crawl', action='store_true',
                       help='Run web crawling to update vector DB')
    parser.add_argument('--skip_rag', action='store_true',
                       help='Skip RAG module (use existing vector DB)')
    parser.add_argument('--skip_multiagent', action='store_true',
                       help='Skip multi-agent module (use existing prompts)')
    parser.add_argument('--skip_synthesis', action='store_true',
                       help='Skip sample synthesis (use existing samples)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip meta-learning training')
    
    args = parser.parse_args()
    config = load_config(args.config)
    if args.pdf_folder:
        config['pdf_folder'] = args.pdf_folder
    
    pipeline = DeepfakeDetectionPipeline(config)
    
    # Run pipeline modules
    if not args.skip_rag:
        pipeline.run_rag_module(
            pdf_folder=args.pdf_folder or config.get('pdf_folder'),
            run_web_crawl=args.run_web_crawl
        )
    
    if not args.skip_multiagent:
        prompts = pipeline.run_multiagent_module(args.user_query)
    else:
        print("Skipping multi-agent module (using existing prompts)")
    
    if not args.skip_synthesis:
        samples = pipeline.run_sample_synthesis_module(args.user_query, args.base_image_dir)
    else:
        print("Skipping sample synthesis (using existing samples)")
    
    if not args.skip_training:
        model, metrics = pipeline.run_meta_learning_module()
    else:
        print("Skipping meta-learning training")
    
    print("\nAll requested modules completed!")


if __name__ == "__main__":
    main()

