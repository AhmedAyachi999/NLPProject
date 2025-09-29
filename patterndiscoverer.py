import pandas as pd
import re
from typing import List, Dict
from openai import OpenAI
from neo4j import GraphDatabase
import numpy as np

# Configure your Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "passwordExample"

# Configure your DeepSeek API key
DEEPSEEK_API_KEY = "APIExample"

# Initialize the Neo4j driver and DeepSeek client
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

class ErrorColumnRelationshipDiscoverer:
    """
    Uses DeepSeek LLM to discover relationships between columns and error types (1-5).
    Uses the entire dataset with chunking for efficient processing.
    """

    def __init__(self, model: str = "deepseek-chat", chunk_size: int = 200):
        self.model = model
        self.relationship_types = {
            "numeric": ["HAS_ABNORMAL_VALUE", "EXCEEDS_THRESHOLD", "BELOW_THRESHOLD",
                        "OUT_OF_RANGE", "HAS_EXTREME_VALUE"],
            "text": ["CONTAINS_PATTERN", "HAS_SPECIFIC_FORMAT", "MISSING_REQUIRED_PATTERN",
                     "HAS_ANOMALOUS_TEXT", "MATCHES_ERROR_PATTERN"],
            "general": ["CORRELATES_WITH", "PREDICTS", "ASSOCIATED_WITH",
                        "INDICATES", "RELATED_TO"]
        }
        self.chunk_size = chunk_size

    def create_data_summary(self, df: pd.DataFrame, error_type: int) -> str:
        """
        Create a comprehensive summary of the data for the LLM analysis.
        Uses statistical summaries and pattern analysis instead of raw data.
        """
        # Filter data for this error type and non-error records
        error_data = df[df['label'] == error_type]
        non_error_data = df[df['label'] == 0]

        summary = f"Analysis for Error Type {error_type}\n"
        summary += "=" * 50 + "\n\n"

        summary += f"Number of Error Type {error_type} records: {len(error_data)}\n"
        summary += f"Number of Non-Error records: {len(non_error_data)}\n\n"

        # Add statistical summaries for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numeric_columns:
            numeric_columns.remove('label')

        summary += "Statistical Summary (Error vs Non-Error):\n"
        summary += "-" * 40 + "\n"

        for col in numeric_columns:
            if col in df.columns:
                error_mean = error_data[col].mean()
                non_error_mean = non_error_data[col].mean()
                error_std = error_data[col].std()
                non_error_std = non_error_data[col].std()

                summary += f"{col}: Error(mean={error_mean:.2f}, std={error_std:.2f}) vs Non-Error(mean={non_error_mean:.2f}, std={non_error_std:.2f})\n"

        summary += "\n"

        # Add pattern analysis for text columns
        text_columns = ['AVO-Kurztext', 'AVO-Langtext', 'VD-Kurztext', 'Beschreibung', 'Langcode', 'Kurzcode']

        summary += "Text Pattern Analysis (Error vs Non-Error):\n"
        summary += "-" * 40 + "\n"

        for col in text_columns:
            if col in df.columns:
                # Get most common values in error data
                error_top = error_data[col].value_counts().head(3)
                non_error_top = non_error_data[col].value_counts().head(3)

                summary += f"{col} - Error top values: {', '.join([f'{k}({v})' for k, v in error_top.items()])}\n"
                summary += f"{col} - Non-Error top values: {', '.join([f'{k}({v})' for k, v in non_error_top.items()])}\n\n"

        return summary

    def analyze_data_chunks(self, df: pd.DataFrame, error_type: int) -> List[Dict]:
        """
        Analyze the data in chunks to discover relationships.
        """
        all_relationships = []

        # Process data in chunks
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size]

            # Check if this chunk contains the error type we're analyzing
            if error_type not in chunk['label'].values:
                continue

            relationships = self.discover_error_relationships(chunk, error_type)
            all_relationships.extend(relationships)

            print(f"Processed chunk {i//self.chunk_size + 1}/{(len(df)//self.chunk_size)+1}, found {len(relationships)} relationships")

        return all_relationships

    def discover_error_relationships(self, df: pd.DataFrame, error_type: int) -> List[Dict]:
        """
        Use DeepSeek LLM to discover relationships between columns and a specific error type.
        """
        # Check if we have enough data for this error type
        error_count = len(df[df['label'] == error_type])
        if error_count < 3:  # Need at least 3 examples
            return []

        # Create data summary instead of using raw data
        data_summary = self.create_data_summary(df, error_type)

        prompt = f"""
        Analyze this data summary and identify the 1-3 MOST IMPORTANT relationships between specific columns and error type {error_type}.
        
        Available columns in the dataset:
        - Lesezeichen: Bookmark/identifier
        - Arbeitsvorgang: Work process/operation
        - AVO-Kurztext: Short text description of work process
        - AVO-Langtext: Long text description of work process  
        - Vorgänge: Processes count
        - Teile pro Vorgang: Parts per process
        - VD-Wert: Value measurement
        - VD-Kurztext: Short text description of value
        - label: Error indicator (0 = correct operation, 1-5 = different error types)
        - Lfd.Nr: Sequential number
        - Beschreibung: Description
        - Langcode: Long code
        - Kurzcode: Short code
        - tg: Time parameter
        - A: Parameter A
        - H: Parameter H  
        - T: Parameter T
        - n (Wert): n value
        - tg*n: Product of tg and n
        
        Error types in the label column:
        - 0: Correct operation (no error)
        - 1: Error type 1
        - 2: Error type 2  
        - 3: Error type 3
        - 4: Error type 4
        - 5: Error type 5
        
        Focus only on the strongest, most significant relationships between columns and error type {error_type}. Look for:
        1. Clear, consistent patterns that strongly differentiate error type {error_type} from correct operations (label 0)
        2. Relationships with high predictive power for this specific error type
        
        For each relationship, provide:
        - error from 0 to 5 0 correct and 1 to 5 different type of errors
        - A meaningful relationship type (choose from: {', '.join(self.relationship_types['general'])})
        - with a column in the dataset. For example here AVO-KURZtext : INDICATES: AVO-Kurztext: Error type 5 is strongly associated with "LL KUP" (Lamellenkupplung) operations, particularly "Lamellenkupplung mit HG picken" which appears in 48% of error cases, while non-error operations are dominated by "Kardanwelle" operations (confidence: 0.88)
        Data Summary:
        {data_summary}
        
        Please provide only the 1-3 most important relationships in this format:
        RELATIONSHIP_TYPE: COLUMN_NAME: PATTERN_DESCRIPTION (CONFIDENCE: X.XX)
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Find even weak patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5
            )

            # Parse the response to extract relationships
            relationships = self._parse_relationships(response.choices[0].message.content, error_type)
            return relationships

        except Exception as e:
            print(f"Error calling DeepSeek LLM for error type {error_type}: {e}")
            return []

    def _parse_relationships(self, response: str, error_type: int) -> List[Dict]:
        """
        Parse the LLM response to extract relationships and confidence scores.
        Now includes relationship type and validates column names.
        """
        relationships = []
        lines = response.split('\n')

        # List of valid column names from your dataset
        valid_columns = [
            'Lesezeichen', 'Arbeitsvorgang', 'AVO-Kurztext', 'AVO-Langtext',
            'Vorgänge', 'Teile pro Vorgang', 'VD-Wert', 'VD-Kurztext', 'label',
            'Lfd.Nr', 'Beschreibung', 'Langcode', 'Kurzcode', 'tg', 'A', 'H', 'T',
            'n (Wert)', 'tg*n'
        ]

        # Patterns to handle different response formats
        patterns = [
            # Primary pattern: RELATIONSHIP_TYPE: COLUMN_NAME: DESCRIPTION (confidence: X.XX)
            r'^([A-Za-z_]+):\s*([^:]+):\s*(.*?)\s*\([Cc]onfidence:\s*([0-9.]+)\)',
            # Fallback pattern: COLUMN_NAME: DESCRIPTION (confidence: X.XX) - without relationship type
            r'^([^:]+):\s*(.*?)\s*\([Cc]onfidence:\s*([0-9.]+)\)',
            # Original patterns for backward compatibility
            r'^([A-Za-z0-9_*]+):\s*(.*?)\s*\(CONFIDENCE:\s*([0-9.]+)\)',
            r'^([A-Za-z0-9_*]+)[:\-]\s*(.*?)\s*\[Confidence:\s*([0-9.]+)\]',
            r'^([A-Za-z0-9_*]+)\s*-\s*(.*?)\s*\(([0-9.]+)\)',
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 4:
                        # Pattern with relationship type
                        relationship_type = groups[0].strip()
                        column_name = groups[1].strip()
                        description = groups[2].strip()
                        confidence = float(groups[3])
                    elif len(groups) == 3:
                        # Pattern without relationship type
                        relationship_type = "RELATED_TO"  # Default value
                        column_name = groups[0].strip()
                        description = groups[1].strip()
                        confidence = float(groups[2])
                    else:
                        continue

                    # Validate column name
                    if column_name not in valid_columns:
                        print(f"Warning: Invalid column name '{column_name}' in line: {line}")
                        continue

                    relationships.append({
                        'error_type': error_type,
                        'relationship_type': relationship_type,
                        'column': column_name,
                        'description': description,
                        'confidence': confidence
                    })
                    break

        return relationships

def store_relationships_in_neo4j(relationships):
    """
    Store the discovered relationships in Neo4j as a knowledge graph.
    Now uses dynamic relationship types from the parsed response.
    """
    if not relationships:
        print("No relationships to store in Neo4j.")
        return

    with driver.session() as session:
        # Clear existing data (optional - comment out if you want to keep existing data)
        # session.run("MATCH (n) DETACH DELETE n")

        # Create column nodes
        unique_columns = set(rel['column'] for rel in relationships)
        for column in unique_columns:
            session.run(
                "MERGE (c:Column {name: $name})",
                name=column
            )

        # Create error type nodes and relationships
        for rel in relationships:
            # Create error type node
            session.run(
                "MERGE (e:ErrorType {type: $type})",
                type=rel['error_type']
            )

            # Sanitize relationship type for Neo4j (replace spaces with underscores)
            rel_type = rel['relationship_type'].replace(' ', '_')

            # Create relationship between error type and column with dynamic type
            query = f"""
            MATCH (e:ErrorType {{type: $error_type}}), (c:Column {{name: $column}})
            MERGE (e)-[r:{rel_type} {{
                description: $description,
                confidence: $confidence
            }}]->(c)
            """
            session.run(
                query,
                error_type=rel['error_type'],
                column=rel['column'],
                description=rel['description'],
                confidence=rel['confidence']
            )

        print(f"Stored {len(relationships)} relationships in Neo4j knowledge graph")

def check_data_distribution(df):
    """
    Check the distribution of error types in the dataset.
    """
    print("Data Distribution Analysis:")
    print("==========================")
    for error_type in range(0, 6):  # 0-5
        count = len(df[df['label'] == error_type])
        print(f"Label {error_type}: {count} records")

    # Check if we have enough data for each error type
    for error_type in range(1, 6):
        count = len(df[df['label'] == error_type])
        if count < 5:
            print(f"Warning: Only {count} records for error type {error_type} - may not find patterns")

def store_relationships_in_neo4j(relationships):

    with driver.session() as session:
        # Clear existing data (optional - you might want to keep existing data)
        # session.run("MATCH (n) DETACH DELETE n")

        # Create column nodes
        unique_columns = set(rel['column'] for rel in relationships)
        for column in unique_columns:
            session.run(
                "MERGE (c:Column {name: $name})",
                name=column
            )

        # Create error type nodes and relationships
        for rel in relationships:
            # Create error type node
            session.run(
                "MERGE (e:ErrorType {type: $type})",
                type=rel['error_type']
            )

            # Create relationship between error type and column with specific type
            session.run(
                """
                MATCH (e:ErrorType {type: $error_type}), (c:Column {name: $column})
                MERGE (e)-[r:%s {
                    description: $description,
                    confidence: $confidence
                }]->(c)
                """ % rel['relationship_type'],
                error_type=rel['error_type'],
                column=rel['column'],
                description=rel['description'],
                confidence=rel['confidence']
            )

        print(f"Stored {len(relationships)} relationships in Neo4j knowledge graph")

def main():
    # Load the data
    try:
        df = pd.read_csv('synth_errs.csv', sep=',')
        print(f"Loaded dataset with {len(df)} records")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check data distribution
    check_data_distribution(df)

    # Initialize relationship discoverer with chunking
    discoverer = ErrorColumnRelationshipDiscoverer(chunk_size=200)  # Adjust chunk size as needed

    all_relationships = []

    # Analyze relationships for each error type (1-5)
    for error_type in range(1, 6):
        print(f"\nAnalyzing relationships for error type {error_type}...")

        # Use chunking to analyze the entire dataset
        relationships = discoverer.analyze_data_chunks(df, error_type)
        all_relationships.extend(relationships)

        print(f"Found {len(relationships)} relationships for error type {error_type}")
        for rel in relationships:
            print(f"  - {rel['column']}: {rel['description']} (confidence: {rel['confidence']})")

    # Store all relationships in Neo4j
    if all_relationships:
        store_relationships_in_neo4j(all_relationships)
        print("Knowledge graph created successfully!")

        # Print summary
        print("\nKnowledge Graph Summary:")
        print("========================")
        for error_type in range(1, 6):
            error_rels = [r for r in all_relationships if r['error_type'] == error_type]
            print(f"Error type {error_type}: {len(error_rels)} column relationships")
    else:
        print("No relationships found for any error type.")

if __name__ == "__main__":
    main()