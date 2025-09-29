import pandas as pd
import numpy as np
from faker import Faker

# Initialize faker for realistic text generation
fake = Faker()
np.random.seed(42)

# Define base components and patterns from original data
brake_components = [
    "Bremsabschirmblech", "Luftleitelement", "Bremssattel",
    "Bremsdruckleitung", "Kabelbinder", "Schraube", "Kardanwelle"
]

positions = ["links", "rechts", "vorne", "hinten", "unten", "oben"]
actions = ["montieren", "heften", "aufnehmen", "platzieren", "verschrauben", "scannen", "holen"]
tools = ["Akkuschrauber", "HG", "EC-Schrauber", "Zange", "Scanner"]

# Generate synthetic data
n_samples = 800
data = []

for i in range(n_samples):
    component = np.random.choice(brake_components)
    position = np.random.choice(positions)
    action = np.random.choice(actions)
    count = np.random.choice(["", "2x ", "3x ", "4x ", "6x "])
    tool = np.random.choice(tools + [""])

    # AVO-Kurztext (short description)
    avo_kurztext = f"{component}-{position} {count}{action}"
    if tool:
        avo_kurztext += f" mit {tool}"

    # AVO-Langtext (long description)
    avo_langtext = f"- {component} {position} aus Zwischenablage aufnehmen "
    avo_langtext += f"- {count}{action} an Position "
    if tool:
        avo_langtext += f"- {tool} verwenden "
    avo_langtext += f"- Qualitätskontrolle durchführen"

    # VD-Kurztext (value description short)
    vd_kurztext_options = [
        f"{component} an Schwenklager",
        f"{component} an {np.random.choice(['Querlenker', 'Spurstange', 'Dämpferstelze'])}",
        f"Einbau {component}",
        f"Montage {component}"
    ]
    vd_kurztext = np.random.choice(vd_kurztext_options)

    # Beschreibung (description)
    beschreibung_options = [
        f"{component} {position} {action} und positionieren",
        f"{count}{action} von {component} {position}",
        f"Vorbereitung für {action} von {component}",
        f"Abschluss {action} von {component} {position}"
    ]
    beschreibung = np.random.choice(beschreibung_options)

    # Kurzcode (short code)
    kurzcode_options = [
        f"3000{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{np.random.randint(1,6)}....{np.random.randint(1,6)}",
        f"M-{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}.{np.random.randint(1,3)}",
        f"LO-{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}",
        f"SEC",
        f"PT.SEC"
    ]
    kurzcode = np.random.choice(kurzcode_options)

    # Determine label based on logical conditions (similar to original)
    complexity_factor = len(component) + len(position) + len(action)
    has_tool = 1 if tool else 0
    has_count = 1 if count else 0

    # Label assignment logic
    if complexity_factor < 20 and not has_tool and np.random.random() < 0.7:
        label = 0  # Correct - simple operations without tools
    elif has_tool and complexity_factor > 25 and np.random.random() < 0.3:
        label = 1  # Error type 1 - complex operations with tools
    elif "scannen" in action or "holen" in action:
        label = 2 if np.random.random() < 0.4 else 0  # Error type 2 - scanning/fetching operations
    elif has_count and int(count[0]) if count and count[0].isdigit() else 1 > 3:
        label = 3 if np.random.random() < 0.5 else 0  # Error type 3 - multiple operations
    elif "Bremssattel" in component or "Kardanwelle" in component:
        label = 4 if np.random.random() < 0.6 else 0  # Error type 4 - critical components
    else:
        label = 5 if np.random.random() < 0.2 else 0  # Error type 5 - other cases

    data.append({
        'AVO-Kurztext': avo_kurztext,
        'AVO-Langtext': avo_langtext,
        'VD-Kurztext': vd_kurztext,
        'Beschreibung': beschreibung,
        'Kurzcode': kurzcode,
        'label': label
    })

# Create DataFrame
df = pd.DataFrame(data)

# Display statistics
print("Dataset Statistics:")
print(f"Total samples: {len(df)}")
print("\nLabel distribution:")
label_counts = df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples ({count/len(df)*100:.1f}%)")

print("\nSample of generated data:")
print(df.head(15))

# Save to CSV
df.to_csv('synthetic_manufacturing_data.csv', index=False)
print(f"\nDataset saved as 'synthetic_manufacturing_data.csv'")

# Show some examples by label
print("\nExamples by label:")
for label in sorted(df['label'].unique()):
    examples = df[df['label'] == label].head(2)
    print(f"\nLabel {label} examples:")
    for idx, row in examples.iterrows():
        print(f"  - {row['AVO-Kurztext']} | {row['Kurzcode']}")