import os
import random
import numpy as np
import timm
import torch
from PIL import Image, UnidentifiedImageError
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from torchvision import transforms
from collections import defaultdict
#import logging
from collections import defaultdict
#from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time, string, pickle
from torchvision.utils import save_image
import torchattacks 
import warnings
warnings.filterwarnings("ignore")

'''class NoErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.ERROR
# Logging setup
logging.basicConfig(filename="\meta_training_log.txt", level=logging.CRITICAL, format="%(message)s")
logging.getLogger().addFilter(NoErrorFilter())'''

# Meta-learning dataset paths
DEFAULT_TRAIN_REAL_PATH = "./meta_learning_data/train_real_classes"
DEFAULT_TRAIN_FAKE_PATH = "./meta_learning_data/train_fake_classes"
DEFAULT_VAL_REAL_PATH = "./meta_learning_data/val_real_classes"
DEFAULT_VAL_FAKE_PATH = "./meta_learning_data/val_fake_classes"
DEFAULT_TEST_REAL_PATH = "./meta_learning_data/test_real_classes"
DEFAULT_TEST_FAKE_PATH = "./meta_learning_data/test_fake_classes"
DEFAULT_FEW_SHOT_SAMPLES_PATH = "./few_shot_samples"

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sanity check to filter corrupt or non-image files
class CustomImageDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.valid_samples, self.valid_labels = self._filter_valid_images()

    def _filter_valid_images(self):
        valid_samples = []
        valid_labels = []
        for img_path, label in zip(self.samples, self.labels):
            try:
                img = Image.open(img_path).convert("RGB")  
                img.verify() 
                valid_samples.append(img_path)
                valid_labels.append(label)
            except (UnidentifiedImageError, IOError):
                #logging.warning(f"Skipping corrupt or invalid image: {img_path}")
                continue
        return valid_samples, valid_labels

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_path = self.valid_samples[idx]
        label = self.valid_labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Meta-learning model setup using a pretrained CoaT model
class MetaLearningModel(nn.Module):
    def __init__(self, num_classes=32):
        super(MetaLearningModel, self).__init__()
        self.model = timm.create_model('coat_lite_tiny', pretrained=True, num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.model(x)

class ReptileMetaLearner:
    def __init__(self, model, inner_steps=5, meta_lr=0.001, inner_lr=0.01, batch_size=128):
        self.model = model
        self.inner_steps = inner_steps
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        self.lambda1, self.lambda2 = 0.5, 0.5

        # Performance metrics dictionary
        self.performance_metrics = defaultdict(list)

       # Lists for metrics
        #self.train_loss_log = []
        #self.val_loss_log = []
        #self.train_acc_log = []
        #self.val_acc_log = []
        
        #label_encoder = LabelEncoder()
        #label_encoder.fit(class_dirs)
        #real_classes = os.listdir(train_real_path)

    def compute_M_adaptive(self, sample, label):
        # M_adaptive = -(p_y - H(p) + Margin - 2*i_misclassified*(1+Margin))
        logits = self.model(sample.unsqueeze(0))  
        probs = torch.nn.functional.softmax(logits, dim=1)
        p_y = probs[label]
        p_s = max(probs[0:label]+probs[label+1:]) 
        margin = abs(p_y - p_s)
        H_p = -sum([p * torch.log(p) for p in sample['prob']])
        i_misclassified = 1  
        M_adaptive = -(p_y - H_p + margin - 2 * i_misclassified * (1 + margin))
        return M_adaptive
    
    def compute_M_adv(self, sample, label):
        M_adv = -self.compute_M_adaptive(sample, label)
        return M_adv

    def rank_samples(self, samples, labels, mode):
        if mode == 'misclassified':
            return sorted(samples, key=lambda s: self.compute_M_adaptive(s, labels[samples.index(s)]))
        elif mode == 'classified':
            return sorted(samples, key=lambda s: self.compute_M_adv(s, labels[samples.index(s)]), reverse=True)
        
    def contrastive_loss(self, samples, aug_samples, labels, margin=1.0):
        distances = torch.norm(samples - aug_samples, p=2, dim=1)
        loss_similar = torch.tensor(labels) * torch.pow(distances, 2)
        loss_dissimilar = (1 - torch.tensor(labels)) * torch.pow(F.relu(margin - distances), 2)
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss
    
    def compute_TSAC_loss(self, samples, aug_samples, labels, Wt):
        contrastive_loss = 0
        for s, ag in zip(samples, aug_samples):
            contrastive_loss += Wt[samples.index(s)]*self.contrastive_loss(s, ag, labels[samples.index(s)])
        return contrastive_loss / len(samples)
    
    def margin_ranking_loss(samples, adv_samples, margin=1.0):
        distances = torch.norm(samples - adv_samples, p=2, dim=1)
        loss = torch.mean(F.relu(margin - distances))
        return loss
    
    def compute_ADV_loss(self, samples, adv_samples, Wadv):
        margin_ranking_loss = 0
        for s, av in zip(samples, adv_samples):
            margin_ranking_loss += Wadv[samples.index(s)]*self.margin_ranking_loss(s, av)
        return margin_ranking_loss / len(samples)
    
    def unified_loss(self, base_loss, samples, aug_samples, adv_samples, labels, Wt, Wadv):    
        weighted_tsac_loss = self.compute_TSAC_loss(samples, aug_samples, labels, Wt)
        weighted_adv_loss = self.compute_ADV_loss(samples, adv_samples, Wadv)
        unified_loss = base_loss + self.lambda1 * weighted_tsac_loss + self.lambda2 * weighted_adv_loss
        return unified_loss

    def initialize_adaptive_weights(num_samples):
        W_t = torch.ones(num_samples, requires_grad=True)/100
        W_adv = torch.ones(num_samples, requires_grad=True)/100
        return W_t, W_adv

    def update_adaptive_weights(self, W_t, W_adv, unified_loss):
        W_t_grad = torch.autograd.grad(unified_loss, W_t, retain_graph=True)[0]
        W_adv_grad = torch.autograd.grad(unified_loss, W_adv, retain_graph=True)[0]
        W_t = W_t - self.inner_lr * W_t_grad
        W_adv = W_adv - self.inner_lr * W_adv_grad
        return W_t, W_adv
    
    def expand_weights(W, num_new_samples):
        new_weights = torch.ones(num_new_samples, requires_grad=True)/100
        W = torch.cat([W, new_weights], dim=0)
        return W

    def update_dataloader(self, adv_samples, adv_labels, aug_samples, aug_labels, train_task_dataset):
        if len(aug_samples) > 0:
            if isinstance(aug_samples, list):
                aug_samples = torch.stack(aug_samples)
            aug_labels = torch.tensor(aug_labels)
            aug_dataset = TensorDataset(aug_samples, aug_labels)
        else:
            aug_dataset = None
            
        if len(adv_samples) > 0:
            if isinstance(adv_samples, list):
                adv_samples = torch.stack(adv_samples)
            adv_labels = torch.tensor(adv_labels)
            adv_dataset = TensorDataset(adv_samples, adv_labels)
        else:
            adv_dataset = None
        
        datasets = [train_task_dataset]
        if aug_dataset is not None:
            datasets.append(aug_dataset)
        if adv_dataset is not None:
            datasets.append(adv_dataset)
            
        combined_dataset = ConcatDataset(datasets)
        train_loader = DataLoader(combined_dataset, batch_size=min(self.batch_size, len(combined_dataset)), shuffle=True)
        return train_loader
    
    def generate_augmentations(self, samples):
        applied_transformations_log = {}
        output_dir = '.\meta_learning_data\Augmented'
        os.makedirs(output_dir, exist_ok=True)
        augmentation_options = {
        'horizontal_flip': transforms.RandomHorizontalFlip(p=1),
        'vertical_flip': transforms.RandomVerticalFlip(p=1),
        'rotate_30': transforms.RandomRotation(30),
        'rotate_90': transforms.RandomRotation(90),
        'color_jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        'grayscale': transforms.Grayscale(num_output_channels=3),
        'resize_128': transforms.Resize((128, 128)),
        'center_crop': transforms.CenterCrop(224),
        'random_equalize': transforms.RandomEqualize(),
        'random_solarize': transforms.RandomSolarize(threshold=192.0),
        'random_perspective': transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        'aug_mix': transforms.AugMix(),
        'auto_contrast': transforms.RandomAutocontrast(),
        'adjust_sharpness': transforms.RandomAdjustSharpness(sharpness_factor=2),
        'random_invert': transforms.RandomInvert(),
        #'random_posterize': transforms.RandomPosterize(bits=4),
        'elastic_transform': transforms.ElasticTransform(alpha=120.0),
        'random_gaussian_blur': transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        'random_resized_crop': transforms.RandomResizedCrop(size=(32, 32)),
        'random_grayscale': transforms.RandomGrayscale(p=1)
    }
        aug_samples = []
        for sample in samples:
            image = transforms.ToPILImage()(sample)
            chosen_augmentations = random.sample(sorted(augmentation_options.items()), random.randint(1, len(augmentation_options)))
         
            augmented_image = image
            applied_augmentations = []
            for aug_name, aug_func in chosen_augmentations:
                augmented_image = aug_func(augmented_image)
                applied_augmentations.append(aug_name)
          
            augmented_image_path = os.path.join(output_dir, f"aug_{''.join((random.choice(string.ascii_letters + string.digits) for i in range(8)))}")
            augmented_image.save(augmented_image_path)
            aug_samples.append(transforms.ToTensor()(augmented_image))
            applied_transformations_log[augmented_image_path] = applied_augmentations
        '''import pickle
        with open('aug_log.pkl','wb') as f:
            pickle.dump(applied_transformations_log, f)'''
        return torch.tensor(aug_samples)
    
    def adversarial_attack_options(self, image_tensor, model):
        attacks = [
            torchattacks.FGSM(model, eps=8/255),
            torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4),
            torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=4),
            torchattacks.CW(model,  c=1, kappa=0, steps=100, lr=0.01),
            torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=2),
            torchattacks.EOTPGD(model, eps=8/255, alpha=4/255, steps=2),
            torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),
            torchattacks.FFGSM(model, eps=8/255, alpha=10/255),
            torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0),
            torchattacks.PGDL2(model, eps=0.3, alpha=0.01, steps=7),
            torchattacks.DeepFool(model, steps=3, overshoot=0.02),
            torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard'),
            torchattacks.AutoAttack(model, norm='L2', eps=8/255, version='standard'),
            torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.02),
            torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
        ]
        applied_attacks = []
        for attack in random.sample(attacks, random.randint(1, len(attacks))):  
            image_tensor = attack(image_tensor, torch.tensor([0]))  
            applied_attacks.append(attack.__class__.__name__)
        return image_tensor, applied_attacks

    def generate_adversarial_variants(self, samples):
        applied_attacks_log = {}
        output_dir = '.\meta_learning_data\Adversarial'
        os.makedirs(output_dir, exist_ok=True)
        
        adv_samples = []
        for sample in samples:
            image = transforms.ToPILImage()(sample)
            adversarial_image_tensor, applied_attacks = self.adversarial_attack_options(image, self.model)
            
            adversarial_image_path = os.path.join(output_dir, f"adv_{''.join((random.choice(string.ascii_letters + string.digits) for i in range(8)))}")
            save_image(adversarial_image_tensor.squeeze(), adversarial_image_path)
            adv_samples.append(adversarial_image_tensor)
            applied_attacks_log[adversarial_image_path] = applied_attacks

        '''import pickle
        with open('adv_log.pkl','wb') as f:
            pickle.dump(applied_attacks_log, f)'''
        return torch.tensor(adv_samples)




    def meta_train(self, train_real_path, train_fake_path, val_real_path, val_fake_path, 
                  num_meta_epochs, tasks_per_epoch, few_shot_samples_path=None):
        #n_way = random.randint(2,34)
        #k_shot = random.randint(20, 50)
        #real_classes = os.listdir(train_real_path)
        criterion = nn.CrossEntropyLoss()
        class_dirs = os.listdir(train_real_path) + os.listdir(train_fake_path)
        
        # Add few-shot samples to the dataset if provided
        few_shot_class_name = None
        if few_shot_samples_path and os.path.exists(few_shot_samples_path):
            # Create a new class for few-shot samples
            few_shot_class_name = "few_shot_generated"
            class_dirs.append(few_shot_class_name)
            print(f"Added few-shot samples from {few_shot_samples_path} as class '{few_shot_class_name}'")
        
        num_classes = len(class_dirs)
        class_mapping = dict(zip(class_dirs, np.arange(num_classes)))

        def sample_task(data_path1, data_path2):
            n_way = random.randint(2, min(15, num_classes))
            k_shot = random.randint(5, 40)
            random.shuffle(class_dirs)
            selected_classes = random.sample(class_dirs, n_way)

            task_samples = []
            labels = []
            description = f"Task Classes: {selected_classes}"

            for class_name in selected_classes:
                # Handle few-shot samples class
                if class_name == few_shot_class_name and few_shot_samples_path:
                    # Get images from few-shot samples directory
                    images = [os.path.join(few_shot_samples_path, img) 
                             for img in os.listdir(few_shot_samples_path)
                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                else:
                    # Get images from regular class directories
                    class_path = os.path.join(data_path1, class_name) if class_name in os.listdir(data_path1) else os.path.join(data_path2, class_name)
                    if not os.path.exists(class_path):
                        continue
                    images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not images:
                    continue
                    
                selected_images = random.sample(images, min(k_shot, len(images)))
                task_samples.extend(selected_images)
                labels.extend([class_mapping[class_name]] * len(selected_images))
            return task_samples, labels, description, n_way, k_shot, selected_classes
        
        def sample_val_task(selected_classes, data_path1, data_path2):
            val_task_samples = []
            val_labels = []

            for class_name in selected_classes:
                # Handle few-shot samples class
                if class_name == few_shot_class_name and few_shot_samples_path:
                    images = [os.path.join(few_shot_samples_path, img) 
                             for img in os.listdir(few_shot_samples_path)
                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                else:
                    class_path = os.path.join(data_path1, class_name) if class_name in os.listdir(data_path1) else os.path.join(data_path2, class_name)
                    if not os.path.exists(class_path):
                        continue
                    images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not images:
                    continue
                    
                val_selected_images = random.sample(images, min(k_shot, len(images)))
                val_task_samples.extend(val_selected_images)
                val_labels.extend([class_mapping[class_name]] * len(val_selected_images))

            return val_task_samples, val_labels

        # Training loop
        for epoch in tqdm(range(num_meta_epochs), desc="Meta-Epochs"):
            #epoch_log = []
            epoch_start_time = time.time()
            ttl = []
            tvl = []
            tta = []
            tva = []
            
            print(f'Epoch {epoch+1}:')
            for task_num in tqdm(range(tasks_per_epoch), desc=f"Tasks (Epoch {epoch+1})"):
                task_samples, task_labels, task_description, n_way, k_shot, selected_classes = sample_task(train_real_path, train_fake_path)
                #epoch_log.append(f'Task {task_num+1}: {task_description}')
                self.performance_metrics['n_way'].append(n_way)
                self.performance_metrics['k_shot'].append(k_shot)
                print(f'Task {task_num+1}: {n_way}-way {k_shot}-shot; {task_description}')
                Wt, Wadv = self.initialize_adaptive_weights(len(task_samples))

                val_samples, val_labels = sample_val_task(selected_classes, val_real_path, val_fake_path)

                # Initialize weights for inner loop training
                initial_weights = {name: param.clone() for name, param in self.model.named_parameters()}

                # Create datasets and dataloaders
                train_task_dataset = CustomImageDataset(task_samples, task_labels, transform=transform)
                val_task_dataset = CustomImageDataset(val_samples, val_labels, transform=transform)
                train_task_loader = DataLoader(train_task_dataset, batch_size=min(self.batch_size, len(train_task_dataset)), shuffle=True)
                val_task_loader = DataLoader(val_task_dataset, batch_size=min(self.batch_size, len(val_task_dataset)), shuffle=True)

                # Inner loop for task-specific updates
                inner_step_loss = 0
                inner_val_loss = 0
                inner_task_train_acc = 0
                inner_task_val_acc = 0

                itl = []
                ivl = []
                ita = []
                iva = []
                classified, classified_labels, misclassified, misclassified_labels = [], [], [], []
                aug_samples_list, adv_samples_list = [], []

                for step in tqdm(range(self.inner_steps), desc=f"Inner Steps (Task {task_num+1})"):
                    running_loss = 0
                    val_step_loss = 0
                    correct_train = 0
                    correct_val = 0
                    total_train = 0
                    total_val = 0

                    # Train Step
                    for batch_samples, batch_labels in train_task_loader:
                        batch_samples = batch_samples.to(self.model.device)
                        batch_labels = torch.tensor(batch_labels).to(self.model.device)#torch.tensor(label_encoder.fit_transform(batch_labels)).to(self.model.device)
                        outputs = self.model(batch_samples) 
                        #bivt = label_encoder.inverse_transform(batch_labels)
                        blt = torch.where(batch_labels <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in bivt])
                        #oivt = label_encoder.inverse_transform(outputs)
                        #olt = torch.where(outputs <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in oivt])
                        
                        base_loss = criterion(outputs, batch_labels.long())#criterion(olt, blt.long())
                        # Use base loss only for first iteration, then add augmentation/adversarial losses
                        if step == 0 or (len(aug_samples_list) == 0 and len(adv_samples_list) == 0):
                            loss = base_loss
                        else:
                            # Use generated samples for unified loss
                            current_aug = aug_samples_list if len(aug_samples_list) > 0 else torch.tensor([])
                            current_adv = adv_samples_list if len(adv_samples_list) > 0 else torch.tensor([])
                            loss = self.unified_loss(base_loss, batch_samples, current_aug, current_adv, batch_labels, Wt, Wadv)
                        #samples, aug_samples, adv_samples, labels, Wt, Wadv

                        self.inner_optimizer.zero_grad()
                        loss.backward()
                        self.inner_optimizer.step()

                        running_loss += loss.item()

                        # Accuracy tracking
                        _, predicted = torch.max(outputs, 1)
                        #pivt = label_encoder.inverse_transform(predicted)
                        plt = torch.where(predicted <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in pivt])                
                        correct_train += (plt == blt).sum().item()
                        total_train += batch_labels.size(0)
                        for idx, sample in enumerate(batch_samples):
                            if plt[idx] == blt[idx]:
                                classified.append(sample)
                                classified_labels.append(blt[idx].item())
                            else:
                                misclassified.append(sample)
                                misclassified_labels.append(blt[idx].item())

                    # Validation Step
                    with torch.no_grad():
                        for val_samples, val_labels in val_task_loader:
                            val_samples = val_samples.to(self.model.device)
                            val_labels = torch.tensor(val_labels).to(self.model.device)#torch.tensor(label_encoder.fit_transform(val_labels)).to(self.model.device)
                            val_outputs = self.model(val_samples)

                            #vbivt = label_encoder.inverse_transform(val_labels)
                            vblt = torch.where(val_labels <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in vbivt])
                            #voivt = label_encoder.inverse_transform(val_outputs)
                            #volt = torch.where(val_outputs <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in voivt])

                            v_loss = criterion(val_outputs, val_labels.long())
                            val_step_loss += v_loss.item()

                            # Validation Accuracy tracking
                            _, val_predicted = torch.max(val_outputs, 1)
                            #vpivt = label_encoder.inverse_transform(val_predicted)
                            vplt = torch.where(val_predicted <= 8, torch.tensor(0).to(self.model.device), torch.tensor(1).to(self.model.device))#torch.tensor([0 if item in real_classes else 1 for item in vpivt])                  
                            correct_val += (vplt == vblt).sum().item()
                            total_val += val_labels.size(0)

                    repre_adv_samples = self.rank_samples(classified, classified_labels, mode='classified')[0:random.randint(2,5)]
                    repre_aug_samples = self.rank_samples(misclassified, misclassified_labels, mode='misclassified')[0:random.randint(2,5)]
                    adv_labels = classified_labels[0:len(repre_adv_samples)]
                    adv_labels = [1 if i >= 8 else 0 for i in adv_labels]
                    aug_labels = misclassified_labels[0:len(repre_aug_samples)]

                    adv_samples_list = self.generate_adversarial_variants(repre_adv_samples)
                    aug_samples_list = self.generate_augmentations(repre_aug_samples)

                    train_task_loader = self.update_dataloader(adv_samples_list, adv_labels, aug_samples_list, aug_labels, train_task_dataset)
                    Wt, Wadv = self.update_adaptive_weights(Wt, Wadv, loss)
                    Wt = self.expand_weights(Wt, len(train_task_loader))
                    Wadv = self.expand_weights(Wadv, len(train_task_loader))

                    inner_step_loss = running_loss / len(train_task_loader)
                    inner_val_loss = val_step_loss / len(val_task_loader)
                    inner_task_train_acc = correct_train / total_train
                    inner_task_val_acc = correct_val / total_val

                    tqdm.write(f"Inner Step {step+1}/{self.inner_steps} - Inner Step Support Set Loss: {inner_step_loss:.4f}  Inner Step Support Set Acc: {inner_task_train_acc:.4f}  Inner Step Query Set Loss: {inner_val_loss:.4f}  Inner Step Query Set Acc: {inner_task_val_acc:.4f}")

                  
                    self.performance_metrics['inner_step_loss'].append(inner_step_loss)
                    self.performance_metrics['inner_val_loss'].append(inner_val_loss)
                    self.performance_metrics['inner_task_acc'].append(inner_task_train_acc)
                    self.performance_metrics['inner_val_acc'].append(inner_task_val_acc)

                    itl.append(inner_step_loss)
                    ivl.append(inner_val_loss)
                    ita.append(inner_task_train_acc)
                    iva.append(inner_task_val_acc)

                # Meta-update the weights
                updated_weights = {name: param.clone() for name, param in self.model.named_parameters()}
                self.meta_update(initial_weights, updated_weights)

                # Logging losses and accuracies
                avg_task_train_loss = sum(itl) / self.inner_steps
                avg_task_val_loss = sum(ivl) / self.inner_steps
                avg_task_train_acc = sum(ita) / self.inner_steps
                avg_task_val_acc = sum(iva) / self.inner_steps

              
                self.performance_metrics['avg_task_train_loss'].append(avg_task_train_loss)
                self.performance_metrics['avg_task_val_loss'].append(avg_task_val_loss)
                self.performance_metrics['avg_task_train_acc'].append(avg_task_train_acc)
                self.performance_metrics['avg_task_val_acc'].append(avg_task_val_acc)

                ttl.append(avg_task_train_loss)
                tvl.append(avg_task_val_loss)
                tta.append(avg_task_train_acc)
                tva.append(avg_task_val_acc)

                '''self.train_loss_log.append(avg_task_train_loss)
                self.val_loss_log.append(avg_task_val_loss)
                self.train_acc_log.append(avg_task_train_acc)
                self.val_acc_log.append(avg_task_val_acc)'''

                #epoch_log.append(f"Task {task_num+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train Acc: {avg_train_acc:.4f} - Val Acc: {avg_val_acc:.4f}")
                #logging.info(f"Epoch {epoch+1} - Task {task_num+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Acc: {avg_val_acc:.4f}")
                print(f'Task {task_num+1} of epoch {epoch + 1}/{num_meta_epochs}, Avg Support Set Task Loss: {avg_task_train_loss:.4f}, Avg Support Set Task Acc: {avg_task_train_acc:.4f}, Avg Query Set Task Loss: {avg_task_val_loss:.4f}, Avg Query Set Task Acc: {avg_task_val_acc:.4f}')

            epoch_duration = time.time() - epoch_start_time
            #logging.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")
            print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")
            '''with open('E:\DeepFakeFace_dataset\Extracted\\'+f'epoch_{epoch+1}_log.txt','w') as f:
                f.write("\n".join(epoch_log))'''

            epoch_avg_train_loss = sum(ttl) / tasks_per_epoch
            epoch_avg_val_loss = sum(tvl) / tasks_per_epoch
            epoch_avg_train_acc = sum(tta) / tasks_per_epoch
            epoch_avg_val_acc = sum(tva) / tasks_per_epoch

            # Update performance metrics dictionary
            self.performance_metrics['epoch_avg_train_loss'].append(epoch_avg_train_loss)
            self.performance_metrics['epoch_avg_val_loss'].append(epoch_avg_val_loss)
            self.performance_metrics['epoch_avg_train_acc'].append(epoch_avg_train_acc)
            self.performance_metrics['epoch_avg_val_acc'].append(epoch_avg_val_acc)
            
            print(f'{epoch + 1}/{num_meta_epochs} completed, Avg Epoch Train Loss: {epoch_avg_train_loss:.4f}, Avg Epoch Train Acc: {epoch_avg_train_acc:.4f}, Avg Epoch Val Loss: {epoch_avg_val_loss:.4f}, Avg Epoch Val Acc: {epoch_avg_val_acc:.4f}')

    def meta_update(self, initial_weights, updated_weights):
        tasks_per_epoch = 10
        for name, param in self.model.named_parameters():
            param.data = initial_weights[name] + self.meta_lr * (updated_weights[name] - initial_weights[name])/ tasks_per_epoch

    def get_performance_metrics(self):
        return self.performance_metrics


def run_meta_learning_pipeline(
    train_real_path: str = DEFAULT_TRAIN_REAL_PATH,
    train_fake_path: str = DEFAULT_TRAIN_FAKE_PATH,
    val_real_path: str = DEFAULT_VAL_REAL_PATH,
    val_fake_path: str = DEFAULT_VAL_FAKE_PATH,
    few_shot_samples_path: str = None,
    num_meta_epochs: int = 30,
    tasks_per_epoch: int = 10,
    num_classes: int = None
):
    """
    Run the complete meta-learning training pipeline.
    
    Args:
        train_real_path: Path to training real images
        train_fake_path: Path to training fake images
        val_real_path: Path to validation real images
        val_fake_path: Path to validation fake images
        few_shot_samples_path: Path to few-shot generated samples (optional)
        num_meta_epochs: Number of meta-epochs
        tasks_per_epoch: Number of tasks per epoch
        num_classes: Number of classes (will be determined automatically if None)
    
    Returns:
        Trained model and performance metrics
    """
    # Determine number of classes
    if num_classes is None:
        class_dirs = os.listdir(train_real_path) + os.listdir(train_fake_path)
        if few_shot_samples_path and os.path.exists(few_shot_samples_path):
            class_dirs.append("few_shot_generated")
        num_classes = len(class_dirs)
    
    # Initialize model
    model = MetaLearningModel(num_classes=num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')
    meta_learner = ReptileMetaLearner(model)
    
    # Train
    meta_learner.meta_train(
        train_real_path=train_real_path,
        train_fake_path=train_fake_path,
        val_real_path=val_real_path,
        val_fake_path=val_fake_path,
        num_meta_epochs=num_meta_epochs,
        tasks_per_epoch=tasks_per_epoch,
        few_shot_samples_path=few_shot_samples_path
    )
    
    # Get performance metrics
    performance_metrics = meta_learner.get_performance_metrics()
    
    # Print summary
    avg_train_loss = sum(performance_metrics["epoch_avg_train_loss"]) / len(performance_metrics["epoch_avg_train_loss"])
    avg_train_acc = sum(performance_metrics["epoch_avg_train_acc"]) / len(performance_metrics["epoch_avg_train_acc"])
    avg_val_loss = sum(performance_metrics["epoch_avg_val_loss"]) / len(performance_metrics["epoch_avg_val_loss"])
    avg_val_acc = sum(performance_metrics["epoch_avg_val_acc"]) / len(performance_metrics["epoch_avg_val_acc"])
    
    print(f'\n{"="*60}')
    print('Meta-Learning Training Complete!')
    print(f'{"="*60}')
    print(f'Average Train Loss (Support Set): {avg_train_loss:.4f}')
    print(f'Average Train Acc (Support Set): {avg_train_acc:.4f}')
    print(f'Average Val Loss (Query Set): {avg_val_loss:.4f}')
    print(f'Average Val Acc (Query Set): {avg_val_acc:.4f}')
    print(f'{"="*60}\n')
    
    return model, performance_metrics


# Main execution for standalone use
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-learning training pipeline")
    parser.add_argument('--train_real_path', type=str, default=DEFAULT_TRAIN_REAL_PATH,
                       help='Path to training real images')
    parser.add_argument('--train_fake_path', type=str, default=DEFAULT_TRAIN_FAKE_PATH,
                       help='Path to training fake images')
    parser.add_argument('--val_real_path', type=str, default=DEFAULT_VAL_REAL_PATH,
                       help='Path to validation real images')
    parser.add_argument('--val_fake_path', type=str, default=DEFAULT_VAL_FAKE_PATH,
                       help='Path to validation fake images')
    parser.add_argument('--few_shot_samples_path', type=str, default=None,
                       help='Path to few-shot generated samples')
    parser.add_argument('--num_meta_epochs', type=int, default=30,
                       help='Number of meta-epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=10,
                       help='Number of tasks per epoch')
    
    args = parser.parse_args()
    
    model, metrics = run_meta_learning_pipeline(
        train_real_path=args.train_real_path,
        train_fake_path=args.train_fake_path,
        val_real_path=args.val_real_path,
        val_fake_path=args.val_fake_path,
        few_shot_samples_path=args.few_shot_samples_path,
        num_meta_epochs=args.num_meta_epochs,
        tasks_per_epoch=args.tasks_per_epoch
    )
    
    # Save model
    model_save_path = "./trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
