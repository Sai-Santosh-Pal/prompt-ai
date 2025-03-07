import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import logging
import re
import random
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptDataset(Dataset):
    """Dataset for prompt enhancement training."""
    
    def __init__(self, basic_prompts, enhanced_prompts, tokenizer, max_length=128, platform_info=None):
        """
        Initialize the dataset.
        
        Args:
            basic_prompts (list): List of basic input prompts
            enhanced_prompts (list): List of corresponding enhanced prompts
            tokenizer: Tokenizer to use for encoding
            max_length (int): Maximum sequence length
            platform_info (list, optional): List of platform names for each prompt
        """
        self.tokenizer = tokenizer
        self.basic_prompts = basic_prompts
        self.enhanced_prompts = enhanced_prompts
        self.max_length = max_length
        self.platform_info = platform_info
        
    def __len__(self):
        return len(self.basic_prompts)
    
    def __getitem__(self, idx):
        basic_prompt = self.basic_prompts[idx]
        enhanced_prompt = self.enhanced_prompts[idx]
        
        # Add platform info if available
        if self.platform_info is not None:
            platform = self.platform_info[idx]
            basic_prompt = f"Platform: {platform} | Prompt: {basic_prompt}"
        
        # Tokenize inputs
        input_encoding = self.tokenizer(
            basic_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            enhanced_prompt,
            max_length=self.max_length * 2,  # Enhanced prompts are typically longer
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": target_encoding.input_ids.flatten(),
        }


class PromptEnhancerAI:
    """AI model for enhancing basic prompts."""
    
    def __init__(self, model_name="t5-base", device=None):
        """
        Initialize the prompt enhancer model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Add special tokens for different platforms
        special_tokens = {
            "additional_special_tokens": [
                "Platform: midjourney", "Platform: dalle", 
                "Platform: stable_diffusion", "Platform: leonardo"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Sample prompts for various categories to help with cold start
        self.sample_prompts = {
            "portrait": [
                {"basic": "woman portrait", 
                 "enhanced": "portrait of a beautiful woman with striking blue eyes, elegant pose, soft natural light, professional photography, highly detailed, shallow depth of field, studio lighting"},
                {"basic": "man in suit", 
                 "enhanced": "professional portrait of a man in tailored black suit, confident expression, corporate environment, dramatic lighting, high contrast, 8K resolution, sharp focus"},
            ],
            "landscape": [
                {"basic": "mountain lake", 
                 "enhanced": "serene mountain lake at sunset, reflections on still water, surrounded by pine trees, majestic snow-capped peaks, golden hour lighting, panoramic view, photorealistic, 8K HDR"},
                {"basic": "desert", 
                 "enhanced": "vast desert landscape with rolling sand dunes, harsh sunlight creating dramatic shadows, distant mountains, clear blue sky, desolate beauty, cinematic wide angle, hyperrealistic"},
            ],
            "food": [
                {"basic": "pasta dish", 
                 "enhanced": "gourmet pasta dish with fresh herbs and parmesan, steam rising, rich tomato sauce, rustic wooden table, soft natural light from side window, shallow depth of field, food photography, mouthwatering details"},
            ],
            "architecture": [
                {"basic": "skyscraper", 
                 "enhanced": "modern glass skyscraper reaching into blue sky, architectural marvel, sleek design, reflective surfaces, urban context, dramatic perspective from ground, professional photography, highly detailed"},
            ],
            "fantasy": [
                {"basic": "wizard tower", 
                 "enhanced": "ancient wizard tower perched on rocky outcrop, magical blue light emanating from windows, storm clouds gathering, lightning in distance, fantasy landscape, mystical atmosphere, detailed stonework, digital art style, epic scene"},
            ]
        }
        
        # Default parameters for different platforms
        self.platform_defaults = {
            "midjourney": {
                "params": ["--v 6", "--style raw", "--ar 16:9", "--q 2"],
                "avoid": "blurry, distorted, disfigured, low quality"
            },
            "dalle": {
                "emphasis": "highly detailed, photorealistic, 8K",
                "avoid": "blurry, distorted, low quality"
            },
            "stable_diffusion": {
                "params": "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7",
                "avoid": "deformed, blurry, bad anatomy, disfigured, poorly drawn, extra limbs"
            },
            "leonardo": {
                "emphasis": "highly detailed, digital painting, concept art",
                "avoid": "deformed, blurry, bad anatomy, disfigured, poorly drawn"
            }
        }
        
    def prepare_training_data(self, save_path="prompt_data.json"):
        """
        Prepare initial training data from sample prompts.
        
        Args:
            save_path (str): Path to save the prepared data
        
        Returns:
            tuple: (basic_prompts, enhanced_prompts, platforms)
        """
        basic_prompts = []
        enhanced_prompts = []
        platforms = []
        
        # Create variations for each platform
        for category, prompts in self.sample_prompts.items():
            for prompt_pair in prompts:
                basic = prompt_pair["basic"]
                enhanced = prompt_pair["enhanced"]
                
                # Add original pair
                basic_prompts.append(basic)
                enhanced_prompts.append(enhanced)
                platforms.append("general")
                
                # Create platform-specific versions
                for platform in ["midjourney", "dalle", "stable_diffusion", "leonardo"]:
                    basic_prompts.append(basic)
                    
                    # Modify the enhanced prompt for the platform
                    platform_enhanced = enhanced
                    
                    # Add platform-specific parameters
                    if platform == "midjourney":
                        platform_enhanced += f" {' '.join(random.sample(self.platform_defaults[platform]['params'], 2))}"
                        platform_enhanced += f" --no {self.platform_defaults[platform]['avoid']}"
                    elif platform == "stable_diffusion":
                        platform_enhanced += f" {self.platform_defaults[platform]['params']}"
                        platform_enhanced += f" Negative prompt: {self.platform_defaults[platform]['avoid']}"
                    elif platform == "dalle":
                        platform_enhanced += f", {self.platform_defaults[platform]['emphasis']}"
                    elif platform == "leonardo":
                        platform_enhanced += f", {self.platform_defaults[platform]['emphasis']}"
                        
                    enhanced_prompts.append(platform_enhanced)
                    platforms.append(platform)
        
        # Generate more variations for data augmentation
        augmented_basic = []
        augmented_enhanced = []
        augmented_platforms = []
        
        for i in range(len(basic_prompts)):
            # Skip some samples to avoid too much duplication
            if random.random() < 0.7:
                continue
                
            basic = basic_prompts[i]
            enhanced = enhanced_prompts[i]
            platform = platforms[i]
            
            # Add adjectives to basic prompt
            adjectives = ["beautiful", "stunning", "amazing", "impressive", "colorful", "dark"]
            adj = random.choice(adjectives)
            new_basic = f"{adj} {basic}"
            
            # Keep the enhanced prompt the same
            augmented_basic.append(new_basic)
            augmented_enhanced.append(enhanced)
            augmented_platforms.append(platform)
        
        # Add augmented data to original data
        basic_prompts.extend(augmented_basic)
        enhanced_prompts.extend(augmented_enhanced)
        platforms.extend(augmented_platforms)
        
        # Save the data
        data = {
            "basic_prompts": basic_prompts,
            "enhanced_prompts": enhanced_prompts,
            "platforms": platforms
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Prepared {len(basic_prompts)} training samples")
        return basic_prompts, enhanced_prompts, platforms
        
    def load_training_data(self, file_path="prompt_data.json"):
        """
        Load training data from a JSON file.
        
        Args:
            file_path (str): Path to the training data file
            
        Returns:
            tuple: (basic_prompts, enhanced_prompts, platforms)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            basic_prompts = data.get("basic_prompts", [])
            enhanced_prompts = data.get("enhanced_prompts", [])
            platforms = data.get("platforms", [])
            
            logger.info(f"Loaded {len(basic_prompts)} training samples")
            return basic_prompts, enhanced_prompts, platforms
            
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
            return self.prepare_training_data(file_path)
    
    def train(self, basic_prompts, enhanced_prompts, platforms=None, 
              batch_size=8, epochs=3, learning_rate=5e-5, 
              save_path="prompt_enhancer_model"):
        """
        Train the model on prompt data.
        
        Args:
            basic_prompts (list): List of basic prompts
            enhanced_prompts (list): List of enhanced prompts
            platforms (list, optional): List of platforms for each prompt
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            save_path (str): Path to save the model
            
        Returns:
            dict: Training history
        """
        # Create dataset
        dataset = PromptDataset(
            basic_prompts, enhanced_prompts, self.tokenizer, platform_info=platforms
        )
        
        # Split into train and validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            logger.info(f"Epoch {epoch+1} - Average validation loss: {avg_val_loss:.4f}")
        
        # Save the model
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Save training history
        with open(os.path.join(save_path, "history.json"), "w") as f:
            json.dump(history, f)
            
        return history
    
    def generate_prompt(self, basic_prompt, platform="midjourney", max_length=256):
        """
        Generate an enhanced prompt from a basic prompt.
        
        Args:
            basic_prompt (str): The basic prompt to enhance
            platform (str): Target platform (midjourney, dalle, stable_diffusion, leonardo)
            max_length (int): Maximum length of the generated prompt
            
        Returns:
            str: Enhanced prompt
        """
        # Add platform information
        input_text = f"Platform: {platform} | Prompt: {basic_prompt}"
        
        # Tokenize input
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        ).input_ids.to(self.device)
        
        # Generate output
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, 
                max_length=max_length,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode output
        enhanced_prompt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return enhanced_prompt
    
    def add_new_prompt_pair(self, basic_prompt, enhanced_prompt, platform="general", save_path="prompt_data.json"):
        """
        Add a new prompt pair to the training data.
        
        Args:
            basic_prompt (str): Basic prompt
            enhanced_prompt (str): Enhanced prompt
            platform (str): Target platform
            save_path (str): Path to the training data file
            
        Returns:
            bool: True if successful
        """
        try:
            # Load existing data
            basic_prompts, enhanced_prompts, platforms = self.load_training_data(save_path)
            
            # Add new pair
            basic_prompts.append(basic_prompt)
            enhanced_prompts.append(enhanced_prompt)
            platforms.append(platform)
            
            # Save updated data
            data = {
                "basic_prompts": basic_prompts,
                "enhanced_prompts": enhanced_prompts,
                "platforms": platforms
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Added new prompt pair to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add new prompt pair: {e}")
            return False
    
    def batch_add_prompt_pairs(self, prompt_pairs, save_path="prompt_data.json"):
        """
        Add multiple prompt pairs to the training data.
        
        Args:
            prompt_pairs (list): List of dictionaries with 'basic', 'enhanced', and 'platform' keys
            save_path (str): Path to the training data file
            
        Returns:
            bool: True if successful
        """
        try:
            # Load existing data
            basic_prompts, enhanced_prompts, platforms = self.load_training_data(save_path)
            
            # Add new pairs
            for pair in prompt_pairs:
                basic_prompts.append(pair["basic"])
                enhanced_prompts.append(pair["enhanced"])
                platforms.append(pair.get("platform", "general"))
            
            # Save updated data
            data = {
                "basic_prompts": basic_prompts,
                "enhanced_prompts": enhanced_prompts,
                "platforms": platforms
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Added {len(prompt_pairs)} prompt pairs to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add prompt pairs: {e}")
            return False
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            bool: True if successful
        """
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# Example usage
def example_usage():
    # Initialize the model
    enhancer = PromptEnhancerAI()
    
    # Prepare/load training data
    basic_prompts, enhanced_prompts, platforms = enhancer.load_training_data()
    
    # Train the model
    history = enhancer.train(basic_prompts, enhanced_prompts, platforms, epochs=2)
    
    # Generate enhanced prompts
    examples = [
        {"basic": "mountain view", "platform": "midjourney"},
        {"basic": "woman portrait", "platform": "dalle"},
        {"basic": "cyberpunk city", "platform": "stable_diffusion"},
        {"basic": "fantasy castle", "platform": "leonardo"}
    ]
    
    for example in examples:
        enhanced = enhancer.generate_prompt(example["basic"], example["platform"])
        print(f"\nBasic: {example['basic']}")
        print(f"Platform: {example['platform']}")
        print(f"Enhanced: {enhanced}")


# Web API for the prompt generator using FastAPI
def create_api():
    try:
        from fastapi import FastAPI, Body, HTTPException
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="Prompt Enhancer API", description="API for enhancing prompts for AI image generation")
        
        # Initialize the model
        enhancer = PromptEnhancerAI()
        # Try to load a trained model, or prepare training data
        try:
            enhancer.load_model("prompt_enhancer_model")
        except Exception:
            basic_prompts, enhanced_prompts, platforms = enhancer.load_training_data()
        
        class PromptRequest(BaseModel):
            prompt: str
            platform: str = "midjourney"
            
        class PromptPair(BaseModel):
            basic: str
            enhanced: str
            platform: str = "general"
            
        @app.post("/enhance")
        async def enhance_prompt(request: PromptRequest):
            """Enhance a basic prompt for a specific platform."""
            try:
                enhanced = enhancer.generate_prompt(request.prompt, request.platform)
                return {"original": request.prompt, "enhanced": enhanced, "platform": request.platform}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.post("/add-prompt")
        async def add_prompt(prompt_pair: PromptPair):
            """Add a new prompt pair to the training data."""
            success = enhancer.add_new_prompt_pair(
                prompt_pair.basic, prompt_pair.enhanced, prompt_pair.platform
            )
            if success:
                return {"status": "success", "message": "Prompt pair added successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to add prompt pair")
                
        @app.post("/batch-add-prompts")
        async def batch_add_prompts(prompt_pairs: list[PromptPair]):
            """Add multiple prompt pairs to the training data."""
            pairs = [{"basic": pair.basic, "enhanced": pair.enhanced, "platform": pair.platform} 
                    for pair in prompt_pairs]
            success = enhancer.batch_add_prompt_pairs(pairs)
            if success:
                return {"status": "success", "message": f"Added {len(pairs)} prompt pairs"}
            else:
                raise HTTPException(status_code=500, detail="Failed to add prompt pairs")
                
        @app.post("/train")
        async def train_model():
            """Train the model on the current training data."""
            try:
                basic_prompts, enhanced_prompts, platforms = enhancer.load_training_data()
                history = enhancer.train(basic_prompts, enhanced_prompts, platforms, epochs=1)
                return {"status": "success", "message": "Model trained successfully", "history": history}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        # Run the API server
        return app
        
    except ImportError:
        logger.error("FastAPI and/or uvicorn not installed. Run 'pip install fastapi uvicorn' to use the API.")
        return None


if __name__ == "__main__":
    example_usage()
    
    # Uncomment to run the API server
    # app = create_api()
    # if app:
    #     import uvicorn
    #     uvicorn.run(app, host="0.0.0.0", port=8000)

enhancer = PromptEnhancerAI()
basic_prompts, enhanced_prompts, platforms = enhancer.prepare_training_data()

enhancer.train(basic_prompts, enhanced_prompts, platforms, epochs=20)

enhanced = enhancer.generate_prompt("mountain view", "midjourney")
print(enhanced)
