import pandas as pd

# Check what the training data looks like
df = pd.read_csv('output/feature_dataset.csv')
print('Training data labels:')
print(df[['label', 'arousal', 'cognitive_load']].value_counts())
print()

print('Amusement samples:')
amusement = df[df['label'] == 'amusement']
print(f'Count: {len(amusement)}')
print(f'Arousal: {amusement["arousal"].unique()}')
print(f'Cognitive Load: {amusement["cognitive_load"].unique()}')
print()

print('Baseline samples:')
baseline = df[df['label'] == 'baseline']
print(f'Count: {len(baseline)}')
print(f'Arousal: {baseline["arousal"].unique()}')
print(f'Cognitive Load: {baseline["cognitive_load"].unique()}')
print()

print('Stress samples:')
stress = df[df['label'] == 'stress']
print(f'Count: {len(stress)}')
print(f'Arousal: {stress["arousal"].unique()}')
print(f'Cognitive Load: {stress["cognitive_load"].unique()}')

# Check amusement feature statistics
print('\n\nAmusement feature statistics:')
print(amusement[['eda_mean', 'eda_std', 'heart_rate_bpm', 'rmssd']].describe())
