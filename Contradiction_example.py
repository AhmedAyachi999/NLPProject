from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class ContradictionDetector:
    """
    BART-large-mNLI model for contradiction detection.
    """

    def __init__(self, model_name="facebook/bart-large-mnli", max_length=512, device=None):
        """
        Initialize the BART-mNLI model.

        Args:
            model_name (str): Hugging Face model name
            max_length (int): Maximum sequence length
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """Load the BART-mNLI model and tokenizer."""
        try:
            print("Loading BART-mNLI model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.is_loaded = True
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def detect_contradiction(self, premise, hypothesis):
        """
        Detect contradiction between two statements.

        Args:
            premise (str): First statement
            hypothesis (str): Second statement to check against premise

        Returns:
            dict: Contradiction analysis results
        """
        if not self.is_loaded:
            self.load_model()

        # Tokenize the premise-hypothesis pair
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)

        # BART-mNLI output indices: 0=contradiction, 1=neutral, 2=entailment
        contradiction_prob = probs[0][0].item()
        neutral_prob = probs[0][1].item()
        entailment_prob = probs[0][2].item()

        # Determine the relationship
        max_prob = max(contradiction_prob, neutral_prob, entailment_prob)
        if max_prob == contradiction_prob:
            relationship = "CONTRADICTION"
        elif max_prob == entailment_prob:
            relationship = "ENTAILMENT"
        else:
            relationship = "NEUTRAL"

        return {
            'relationship': relationship,
            'contradiction_prob': contradiction_prob,
            'neutral_prob': neutral_prob,
            'entailment_prob': entailment_prob,
            'premise': premise,
            'hypothesis': hypothesis
        }

    def print_detailed_result(self, result):
        """Print detailed contradiction analysis results."""
        print("\n" + "="*60)
        print("CONTRADICTION ANALYSIS RESULTS")
        print("="*60)
        print(f"Premise:      {result['premise']}")
        print(f"Hypothesis:   {result['hypothesis']}")
        print(f"Relationship: {result['relationship']}")
        print("\nProbabilities:")
        print(f"  Contradiction: {result['contradiction_prob']:.4f} ({result['contradiction_prob']*100:.2f}%)")
        print(f"  Neutral:       {result['neutral_prob']:.4f} ({result['neutral_prob']*100:.2f}%)")
        print(f"  Entailment:    {result['entailment_prob']:.4f} ({result['entailment_prob']*100:.2f}%)")
        print("="*60)

def main():
    """Main method for manual contradiction detection."""
    print("BART-mNLI Contradiction Detector")
    print("This model detects contradictions, entailments, and neutral relationships between sentences.")

    # Initialize the detector
    detector = ContradictionDetector()

    while True:
        print("\n" + "-"*50)
        print("Enter two sentences to analyze their relationship:")
        print("-"*50)

        # Get user input
        premise = input("\nEnter the first sentence (premise): ").strip()
        if not premise:
            print("No premise entered. Exiting...")
            break

        hypothesis = input("Enter the second sentence (hypothesis): ").strip()
        if not hypothesis:
            print("No hypothesis entered. Exiting...")
            break

        try:
            # Detect contradiction
            print("\nAnalyzing...")
            result = detector.detect_contradiction(premise, hypothesis)

            # Display results
            detector.print_detailed_result(result)

            # Simple interpretation
            print("\nInterpretation:")
            if result['relationship'] == 'CONTRADICTION':
                print("❌ The two sentences CONTRADICT each other.")
            elif result['relationship'] == 'ENTAILMENT':
                print("✅ The hypothesis is ENTAILED by the premise.")
            else:
                print("➖ The relationship is NEUTRAL (neither contradiction nor entailment).")

        except Exception as e:
            print(f"Error during analysis: {e}")

        # Ask if user wants to continue
        while True:
            continue_choice = input("\nAnalyze another pair? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes', 'n', 'no']:
                break
            print("Please enter 'y' for yes or 'n' for no.")

        if continue_choice in ['n', 'no']:
            print("Thank you for using the Contradiction Detector!")
            break

if __name__ == "__main__":
    # Run examples first
    example_usage()

    # Then start interactive mode
    main()