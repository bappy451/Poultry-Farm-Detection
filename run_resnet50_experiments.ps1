# Run SparK SSL Pretrained Experiment
Write-Host "Starting ResNet50 SparK Pretrained Fine-tuning..."
python src/downstream_finetune/finetune_resnet50.py --data Data/downstream_data.yaml --weights trained_models/e15841a515ed_resnet50_fp16/e15841a515ed_resnet50_fp16/backbone_dense_ep7.pt --project runs/segment/resnet50_spark_finetune --epochs 50 --batch 4
Write-Host "Finished ResNet50 SparK Pretrained Fine-tuning."

# Run ImageNet Pretrained Experiment
Write-Host "Starting ResNet50 ImageNet Pretrained Fine-tuning..."
python src/downstream_finetune/finetune_resnet50.py --data Data/downstream_data.yaml --project runs/segment/resnet50_pretrained_finetune --epochs 50 --batch 4 --pretrained
Write-Host "Finished ResNet50 ImageNet Pretrained Fine-tuning."

# Run Random Initialization Experiment (Non-trained baseline)
Write-Host "Starting ResNet50 Random Initialization Fine-tuning..."
python src/downstream_finetune/finetune_resnet50.py --data Data/downstream_data.yaml --project runs/segment/resnet50_random_finetune --epochs 50 --batch 4
Write-Host "Finished ResNet50 Random Initialization Fine-tuning."
