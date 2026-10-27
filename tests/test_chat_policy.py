import unittest

from src.api.chat_policy import get_retrieval_mode, is_medical_advice_request


class ChatPolicyTests(unittest.TestCase):
    def test_blocks_requests_for_treatment_recommendations(self):
        questions = (
            "Should I prescribe a higher dose of Furosemide?",
            "Which antibiotic should I start?",
            "Can you prescribe me a new medication?",
            "What is the best treatment for fluid retention?",
            "Diagnose my symptoms based on this chart.",
            "Should the patient increase the dosage?",
            "Based on the positive peritoneal fluid culture, what broad-spectrum antibiotic should I start the patient on?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_medical_advice_request(question))

    def test_allows_patient_history_questions(self):
        questions = (
            "What medications were prescribed to this patient upon discharge?",
            "What liver-related diagnoses are noted in the patient's file?",
            "Was the patient admitted urgently or routinely?",
            "Can you summarise the last few reports of this patient?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(is_medical_advice_request(question))
        self.assertFalse(is_medical_advice_request("Should I change the appointment date?"))

    def test_selects_targeted_retrieval_modes(self):
        self.assertEqual(
            get_retrieval_mode("Was the patient admitted urgently or routinely?"),
            "admission",
        )
        self.assertEqual(
            get_retrieval_mode("What medications were prescribed upon discharge?"),
            "medications",
        )
        self.assertEqual(
            get_retrieval_mode("Summarise the last few reports."),
            "reports",
        )
        self.assertEqual(get_retrieval_mode("What was the Furosemide dosage?"), "medications")
        self.assertEqual(get_retrieval_mode("What is the patient's history of fluid retention?"), "semantic")


if __name__ == "__main__":
    unittest.main()
