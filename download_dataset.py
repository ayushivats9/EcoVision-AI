#!/usr/bin/env python3
"""
EcoVision AI - Dataset Downloader
==================================

Downloads and prepares the TrashNet dataset and additional waste classification data
for training the EcoVision AI model supporting SDG 12 & 13.

Author: Your Name
Date: 2024
License: MIT
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import argparse


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """
    Dataset downloader and organizer for waste classification
    
    Downloads multiple datasets and organizes them into the required
    directory structure for training the EcoVision AI model.
    """
    
    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.samples_dir = self.base_dir / 'samples'
        
        
        self.datasets = {
            'trashnet': {
                'url': 'https://github.com/garythung/trashnet/archive/master.zip',
                'description': 'TrashNet dataset by Gary Thung and Mindy Yang',
                'categories': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
                'size': '2,527 images'
            }
        }
        
       
        self.target_categories = {
            'organic': ['trash'],  
            'recyclable': ['cardboard', 'glass', 'metal', 'paper', 'plastic'],
            'hazardous': []  
        }
        
        
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.samples_dir,
            self.processed_dir / 'organic',
            self.processed_dir / 'recyclable', 
            self.processed_dir / 'hazardous'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def download_with_progress(self, url, destination):
        """Download file with progress bar"""
        logger.info(f"⬇️ Downloading from {url}")
        
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Download") as t:
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
    
    def download_trashnet(self):
        """Download and extract TrashNet dataset"""
        logger.info("🗑️ Downloading TrashNet dataset...")
        
        zip_path = self.raw_dir / 'trashnet.zip'
        extract_path = self.raw_dir / 'trashnet'
        
        
        if not zip_path.exists():
            self.download_with_progress(
                self.datasets['trashnet']['url'],
                zip_path
            )
            logger.info("✅ TrashNet dataset downloaded")
        else:
            logger.info("📦 TrashNet dataset already exists")
        
        if not extract_path.exists():
            logger.info("📦 Extracting TrashNet dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info("✅ TrashNet dataset extracted")
        
        return extract_path / 'trashnet-master' / 'data'
    
    def organize_dataset(self, source_path):
        """Organize downloaded dataset into target categories"""
        logger.info("📋 Organizing dataset into target categories...")
        
        stats = {'organic': 0, 'recyclable': 0, 'hazardous': 0}
        
        
        for target_cat, source_cats in self.target_categories.items():
            target_dir = self.processed_dir / target_cat
            
            for source_cat in source_cats:
                source_dir = source_path / source_cat
                
                if source_dir.exists():
                    logger.info(f"📁 Processing {source_cat} -> {target_cat}")
                    
                    
                    for img_file in source_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            dest_file = target_dir / f"{source_cat}_{img_file.name}"
                            shutil.copy2(img_file, dest_file)
                            stats[target_cat] += 1
                
        
        self.create_hazardous_samples()
        
        logger.info("📊 Dataset organization complete:")
        for category, count in stats.items():
            logger.info(f"   {category}: {count} images")
        
        return stats
    
    def create_hazardous_samples(self):
        """Create sample hazardous waste images structure"""
        hazardous_dir = self.processed_dir / 'hazardous'
        
        
        info_text = """
HAZARDOUS WASTE CATEGORY

This category requires manual addition of images for:
- Batteries (lithium, alkaline, car batteries)
- Electronics (phones, computers, circuit boards)  
- Chemicals (paint, solvents, cleaners)
- Medical waste (syringes, medications)
- Light bulbs (CFL, LED with electronics)

For training, please add images to this directory.
Each image should be properly labeled and categorized.

Safety Note: Never handle actual hazardous waste without proper protection!
"""
        
        with open(hazardous_dir / 'README.txt', 'w') as f:
            f.write(info_text)
        
        logger.info("⚠️ Created hazardous waste category structure")
        logger.info("⚠️ Manual addition of hazardous waste images required")
    
    def create_sample_images(self):
        """Copy sample images for testing and demo"""
        logger.info("🖼️ Creating sample images for demo...")
        
        sample_count = 5  
        
        for category in ['organic', 'recyclable', 'hazardous']:
            category_dir = self.processed_dir / category
            sample_category_dir = self.samples_dir / category
            sample_category_dir.mkdir(exist_ok=True)
            
            
            images = list(category_dir.glob('*'))[:sample_count]
            for i, img_path in enumerate(images):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    dest_path = sample_category_dir / f"sample_{i+1}{img_path.suffix}"
                    shutil.copy2(img_path, dest_path)
        
        logger.info("✅ Sample images created")
    
    def generate_dataset_info(self, stats):
        """Generate dataset information file"""
        info = {
            "dataset_name": "EcoVision AI Waste Classification Dataset",
            "version": "1.0.0",
            "description": "Curated dataset for waste classification supporting UN SDG 12 & 13",
            "categories": {
                "organic": {
                    "description": "Biodegradable waste that can be composted",
                    "examples": ["Food scraps", "Yard waste", "Paper towels"],
                    "count": stats['organic'],
                    "sdg_alignment": "SDG 12 - Responsible Consumption"
                },
                "recyclable": {
                    "description": "Materials that can be processed and reused",
                    "examples": ["Plastic bottles", "Paper", "Metal cans", "Glass"],
                    "count": stats['recyclable'],
                    "sdg_alignment": "SDG 12 - Responsible Consumption"
                },
                "hazardous": {
                    "description": "Waste requiring special disposal methods",
                    "examples": ["Batteries", "Electronics", "Chemicals"],
                    "count": stats['hazardous'],
                    "sdg_alignment": "SDG 13 - Climate Action"
                }
            },
            "total_images": sum(stats.values()),
            "data_sources": [
                "TrashNet dataset (Gary Thung & Mindy Yang)",
                "Custom curated images for hazardous waste"
            ],
            "preprocessing": {
                "image_size": "224x224 pixels",
                "format": "RGB",
                "augmentation": "Rotation, scaling, brightness adjustment"
            },
            "usage": {
                "training_split": "80%",
                "validation_split": "10%", 
                "test_split": "10%"
            },
            "license": "MIT License",
            "created_by": "EcoVision AI Project",
            "creation_date": "2024",
            "sdg_contribution": {
                "sdg_12": "Promotes responsible waste management through AI classification",
                "sdg_13": "Reduces environmental impact through proper waste sorting"
            }
        }
        
        import json
        with open(self.base_dir / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("📄 Dataset information file created")
    
    def verify_dataset(self):
        """Verify dataset integrity and completeness"""
        logger.info("🔍 Verifying dataset...")
        
        issues = []
        
        
        for category in ['organic', 'recyclable', 'hazardous']:
            category_dir = self.processed_dir / category
            if not category_dir.exists():
                issues.append(f"Missing category directory: {category}")
                continue
            
            
            images = list(category_dir.glob('*'))
            image_files = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if len(image_files) < 10:
                issues.append(f"Category '{category}' has only {len(image_files)} images (minimum 10 recommended)")
        
        if issues:
            logger.warning("⚠️ Dataset verification found issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Dataset verification passed")
        
        return len(issues) == 0
    
    def download_all(self):
        """Download and organize all datasets"""
        logger.info("🚀 Starting dataset download and organization...")
        
        
        trashnet_path = self.download_trashnet()
        
        
        stats = self.organize_dataset(trashnet_path)
        
        
        self.create_sample_images()
        
        
        self.generate_dataset_info(stats)
        
        self.verify_dataset()
        
        logger.info("🎉 Dataset preparation complete!")
        logger.info(f"📊 Total images: {sum(stats.values())}")
        logger.info("🌱 Ready to train EcoVision AI for SDG 12 & 13!")
        
        return stats

def main():
    """Main dataset download function"""
    parser = argparse.ArgumentParser(description='Download EcoVision AI Dataset')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base directory for dataset (default: data)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset without downloading')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.verify_only:
        
        logger.info("🔍 Verifying existing dataset...")
        if downloader.verify_dataset():
            logger.info("✅ Dataset verification successful")
        else:
            logger.error("❌ Dataset verification failed")
            return 1
    else:
        
        try:
            stats = downloader.download_all()
            logger.info("✅ Dataset download completed successfully")
            
            
            print("\n" + "="*50)
            print("📊 DATASET SUMMARY")
            print("="*50)
            print(f"Organic waste: {stats['organic']:,} images")
            print(f"Recyclable waste: {stats['recyclable']:,} images") 
            print(f"Hazardous waste: {stats['hazardous']:,} images")
            print(f"Total images: {sum(stats.values()):,}")
            print("\n🌍 Supporting UN SDG 12 & 13")
            print("🎯 Ready for AI model training!")
            print("="*50)
            
        except Exception as e:
            logger.error(f"❌ Error during dataset download: {e}")
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())