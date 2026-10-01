For Problem Statement: 
Causal Rationale Extraction and Synthesis from Conversational Data on Business Events
Note: This dataset should NOT be used for any other purposes than the internal hackathon. It is copyrighted by the company. 
Dataset Properties:
* Total transcripts: 19621 
* All transcripts are in English language
* 2-party turn-by-turn conversation
   * Agent -> Represents a call center agent who is conversing with the customer to support on their issue
   * Customer -> Represents a customer who is calling in to a call center to get their issue or query resolved
* Since data represents real world conversations - you may find noise in data such as ill punctuations, non grammatical segments/phrases, conversational pauses etc.
* Metadata (example):
      * "transcript_id": "ce44aa4b-13b7-4778-b413-15f37ee15f7a"
      * "time_of_interaction": "2025-11-10 16:23:00"
      * "domain": "Insurance"
      * "intent": "Discounts & Promotions"
      * "reason_for_call": "Inquiry about promotional codes and potential loyalty benefits for booking at Beachfront Suites in Las Vegas."
   * To be noted that other than transcript id, other meta-data fields are optional and could be null as well.
Sample conversation (formatted in plain text for easy readability)
Customer: Hi I’m calling about my health claim again. I’ve sent the documents twice already, and it still says “under review”. It’s been almost three weeks.
Agent: I’m really sorry you’ve had to go through this, sir. Let me pull up your file first — could you confirm your policy number for me?
Customer: Yeah, it’s 4561-HCP. Honestly, this has been pretty stressful. Every time I call, I get a different answer.
Agent: I completely understand — that inconsistency can be really frustrating. Just a moment… (pause) Okay, I see what happened. The medical report from your clinic was uploaded, but it was missing a doctor’s signature, so it didn’t pass verification.
Customer: Wait, what? No one told me that. The report was complete when I sent it!
Agent: You’re right — someone should’ve informed you earlier. I can see it was marked for manual review but never updated on your portal. That’s on us. If you’d like, I can send you the exact document version that’s missing the signature — would that help?
Customer: Yeah, please do. I’ll get it fixed right away. But I hope this doesn’t restart the whole process again?
Agent: Not at all. Once you re-upload it, I’ll personally flag the claim for priority processing. That means it’ll be reviewed within 24 hours instead of the usual 48.
Customer: Okay, that’s better. I just wish someone had told me this earlier — I’ve been paying premiums for years, you know?
Agent: Absolutely fair. You’ve been with us since 2018, so your account is in good standing — this delay definitely shouldn’t have happened. I’ll make sure the claim team knows that. Also, just so you’re aware, once it’s approved, the refund goes straight to your linked bank account — you won’t need to follow up again.
Customer: Alright, thanks, Lisa. You’ve actually explained this clearly. I feel a bit more confident now.
Agent: I’m really glad to hear that. I’ll send you a confirmation email in the next 10 minutes, and once you’ve re-uploaded, I’ll keep an eye on it personally. Anything else I can help with today?
Customer: No, that’s it. Appreciate your help — finally feels like someone’s taking ownership.
Agent: Thank you for saying that. We’ll make sure this gets sorted out properly. Have a good day, and take care!