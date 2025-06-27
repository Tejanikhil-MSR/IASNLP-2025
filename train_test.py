import torch
from utils import load_waveform, extract_prosodic_feature
from config import CONFIG

def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        losses = []
        epoch_loss = 0
        correct = 0
        total = 0

        for (batch_audio, valid_frames, batch_prosody, batch_labels) in dataloader:
            audio_tensors = batch_audio.to(device)
            valid_frames = torch.tensor(valid_frames).to(device)
            prosody_tensor = batch_prosody.to(device)
            labels = torch.tensor(batch_labels).to(device)

            optimizer.zero_grad()
            outputs = model(audio_tensors=audio_tensors, valid_frames=valid_frames, prosody_tensor=prosody_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.numel()

            print(epoch_loss)

        losses.append(epoch_loss)

        acc = 100 * correct / total
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.2f}%")

    return model, losses
        
        
def test(audio, model, preprocessor, device):
    f, f_len = load_waveform(audio, preprocessor, CONFIG["max_audio_sequence_length"], device)
    f_pros = extract_prosodic_feature(audio)
    output = model(f, CONFIG["max_audio_sequence_length"] - f_len, f_pros)
    preds = torch.sigmoid(output) > 0.5
    return preds