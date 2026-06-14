"""
scripts/generate_pdf.py

Generates the 7-Day Wellness Protocol PDF that serves as the RAG knowledge base.
Run once before ingestion: python scripts/generate_pdf.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from any cwd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF, XPos, YPos

OUTPUT_PATH = ROOT / "data" / "wellness_protocol.pdf"


PROTOCOL_CONTENT = {
    "title": "7-Day Foundational Wellness Protocol",
    "subtitle": "Your Personal Guide to Building Lasting Health Habits",
    "overview": (
        "This protocol is designed for individuals beginning their wellness journey. "
        "Over seven days you will establish foundational habits across four pillars: "
        "Sleep, Hydration & Nutrition, Movement, and Mental Wellness. "
        "Each day builds on the last, creating a compounding effect that sets the "
        "stage for long-term health transformation.\n\n"
        "Important: This protocol is educational and supportive in nature. "
        "Always consult a qualified healthcare provider before making significant "
        "changes to your diet, exercise, or health routine."
    ),
    "general_rules": [
        "Track every habit daily using your wellness journal or the app check-in feature.",
        "Drink at least 2.5 litres of water per day, starting with a glass upon waking.",
        "Aim for 7–9 hours of sleep each night; consistency of sleep schedule matters more than total hours.",
        "Avoid screens for at least 30 minutes before bed to protect sleep quality.",
        "Eat at least 5 portions of vegetables and fruit daily.",
        "Limit added sugar to fewer than 25 g per day.",
        "Do not skip meals; eat within 1 hour of waking to stabilise blood sugar.",
        "Take 10 deep diaphragmatic breaths every morning before leaving bed.",
        "Journal for at least 5 minutes each evening — note one win and one challenge.",
        "Weigh yourself at most once per week, same time, same conditions.",
    ],
    "dos_and_donts": {
        "dos": [
            "DO start each day with a glass of water before coffee or tea.",
            "DO prioritise protein at every meal (eggs, legumes, meat, dairy, tofu).",
            "DO go outside for at least 15 minutes of natural light each morning.",
            "DO celebrate small wins — habit streaks build momentum.",
            "DO communicate any concerns or questions during your daily check-in.",
            "DO adapt movement intensity to how you feel — rest is productive too.",
            "DO use a consistent bedtime routine: dim lights, no screens, calm activity.",
        ],
        "donts": [
            "DON'T weigh yourself more than once per week — daily fluctuations are misleading.",
            "DON'T skip the evening journal — reflection drives behaviour change.",
            "DON'T attempt extreme caloric restriction during this protocol.",
            "DON'T use this protocol as a substitute for medical advice.",
            "DON'T compare your progress to others — this protocol is personalised for you.",
            "DON'T drink alcohol during the 7-day programme; it disrupts sleep architecture.",
            "DON'T exercise intensely if you slept fewer than 5 hours — opt for a gentle walk.",
        ],
    },
    "tracking_rules": [
        "Log sleep duration and quality (1–5 scale) each morning.",
        "Log water intake in 250 ml increments throughout the day.",
        "Log meals with a brief description — no calorie counting required.",
        "Log movement type and duration after each session.",
        "Log mood (1–10 scale) at noon and at bedtime.",
        "Log your one win and one challenge in the evening journal.",
        "Share a brief check-in summary with your coach every day via the agent.",
        "If you miss a tracking entry, do not skip the next — just continue.",
    ],
    "days": [
        {
            "day": 1,
            "title": "Baseline & Orientation",
            "theme": "Awareness",
            "objectives": [
                "Complete your wellness profile with the coach.",
                "Establish baseline measurements: resting heart rate, sleep quality, energy (1–10).",
                "Drink 2.5 L of water today — set hourly reminders if helpful.",
                "Go for a 20-minute gentle walk, any time of day.",
                "Journal: Write 3 things you want to feel like at the end of this week.",
            ],
            "sleep_guidance": (
                "Tonight, set a consistent bedtime. Aim for 7.5–8 hours. "
                "Put your phone in another room or use night mode 30 minutes before bed."
            ),
            "nutrition_guidance": (
                "No specific meal plan today. Simply observe your current eating patterns "
                "without judgment. Note what you eat and when in your journal."
            ),
            "movement_guidance": "20-minute gentle walk. No intensity target today — just move.",
            "coach_note": (
                "Day 1 is about awareness, not perfection. Notice your habits "
                "without changing them drastically. Curiosity is your best tool today."
            ),
        },
        {
            "day": 2,
            "title": "Sleep Optimisation",
            "theme": "Rest & Recovery",
            "objectives": [
                "Implement a 30-minute wind-down routine tonight.",
                "Avoid caffeine after 2 pm.",
                "Complete your morning gratitude breathing (10 deep breaths).",
                "Hydrate: 2.5 L water minimum.",
                "Add one serving of leafy greens to a meal today.",
            ],
            "sleep_guidance": (
                "Wind-down routine: dim lights at 9 pm, avoid screens, try reading or "
                "stretching. Keep the bedroom cool (16–19°C is optimal for sleep). "
                "Avoid large meals within 2 hours of bedtime."
            ),
            "nutrition_guidance": (
                "Prioritise protein at breakfast (20–30 g). This stabilises blood sugar "
                "and reduces cravings throughout the day. Examples: eggs, Greek yoghurt, "
                "protein smoothie."
            ),
            "movement_guidance": "25-minute walk or gentle yoga. Keep it easy today.",
            "coach_note": (
                "Sleep is the foundation of every other health metric. "
                "One good night's sleep improves mood, focus, metabolism, and willpower."
            ),
        },
        {
            "day": 3,
            "title": "Hydration & Nutrition Focus",
            "theme": "Fuelling Well",
            "objectives": [
                "Drink a large glass of water immediately upon waking.",
                "Eat at least 5 servings of vegetables and fruit.",
                "Limit added sugar to under 25 g today.",
                "Eat within 1 hour of waking.",
                "Identify one processed food to reduce or eliminate.",
            ],
            "sleep_guidance": (
                "Maintain yesterday's wind-down routine. Consistency over 3 nights "
                "begins to shift your circadian rhythm positively."
            ),
            "nutrition_guidance": (
                "Build a balanced plate: 50% vegetables, 25% protein, 25% complex carbohydrates. "
                "Choose whole grains over refined (brown rice, oats, quinoa). "
                "Avoid liquid calories — juice, soda, and energy drinks spike insulin sharply."
            ),
            "movement_guidance": (
                "30-minute walk. Try adding 5 minutes of body-weight strength work: "
                "10 squats, 10 push-ups (modified is fine), 20-second plank."
            ),
            "coach_note": (
                "Hydration affects energy, mood, and cognition more than most people realise. "
                "If you feel a 3 pm slump, drink water before reaching for caffeine."
            ),
        },
        {
            "day": 4,
            "title": "Movement & Strength",
            "theme": "Building Physical Capacity",
            "objectives": [
                "Complete a 30–40 minute movement session of your choice.",
                "Achieve 7,000+ steps today.",
                "Stretch for 10 minutes after movement.",
                "Continue all hydration and sleep habits.",
                "Journal: How has your energy changed since Day 1?",
            ],
            "sleep_guidance": (
                "Exercise improves sleep quality but should be completed at least "
                "3 hours before bedtime to avoid elevated cortisol disrupting sleep onset."
            ),
            "nutrition_guidance": (
                "On higher-movement days, add a complex carbohydrate source at the meal "
                "before your session (oats, sweet potato, banana). "
                "Post-session, consume protein within 45 minutes."
            ),
            "movement_guidance": (
                "Choose an activity you enjoy: brisk walk, cycling, swimming, strength training, "
                "dance class. Duration matters more than intensity at this stage. "
                "Target: moderate effort where you can hold a conversation."
            ),
            "coach_note": (
                "Regular movement is the closest thing to a 'wonder drug' in health research — "
                "it improves sleep, mood, insulin sensitivity, cognitive function, and longevity."
            ),
        },
        {
            "day": 5,
            "title": "Mental Wellness & Stress Management",
            "theme": "Mind & Emotions",
            "objectives": [
                "Complete a 10-minute guided meditation or mindfulness practice.",
                "Identify your top stressor this week and write one small action to address it.",
                "Spend 20 minutes on an activity purely for enjoyment (no productivity).",
                "Practice 4-7-8 breathing before bed: inhale 4s, hold 7s, exhale 8s.",
                "Continue all prior habits.",
            ],
            "sleep_guidance": (
                "Stress is the number-one disruptor of sleep quality. "
                "The 4-7-8 breathing technique activates the parasympathetic nervous system "
                "and significantly reduces time to sleep onset."
            ),
            "nutrition_guidance": (
                "Foods that support mental wellness: oily fish (omega-3), dark leafy greens "
                "(magnesium), fermented foods (gut-brain axis), dark chocolate (70%+, 1–2 squares). "
                "Avoid ultra-processed foods and alcohol, which worsen anxiety and mood stability."
            ),
            "movement_guidance": (
                "Today, a 20-minute walk in nature counts as both movement AND mental wellness. "
                "Research shows green-space exposure reduces cortisol."
            ),
            "coach_note": (
                "Mental health is physical health. Chronic stress raises cortisol, disrupts sleep, "
                "increases appetite for high-calorie foods, and impairs immune function. "
                "Managing stress is non-negotiable in a wellness protocol."
            ),
        },
        {
            "day": 6,
            "title": "Habit Stack & Integration",
            "theme": "Compounding Your Habits",
            "objectives": [
                "Review all five previous days — which habit felt most natural?",
                "Design your personal 'morning habit stack' (3–5 habits chained together).",
                "Complete all daily habits without reminders if possible.",
                "Share your habit stack with your coach during check-in.",
                "Plan your meals for the next day in advance.",
            ],
            "sleep_guidance": (
                "A consistent sleep and wake time — even on weekends — is the single most "
                "powerful sleep intervention. Aim to be within 30 minutes of the same "
                "schedule 7 days a week."
            ),
            "nutrition_guidance": (
                "Meal planning reduces decision fatigue and prevents impulsive food choices. "
                "Prepare at least one component in advance: pre-washed vegetables, "
                "cooked grains, or portioned snacks."
            ),
            "movement_guidance": "35-minute session of your preferred activity from Day 4.",
            "coach_note": (
                "A 'habit stack' links a new habit to an existing one using the formula: "
                "'After I [existing habit], I will [new habit].' "
                "Example: 'After I pour my morning coffee, I will drink a full glass of water first.'"
            ),
        },
        {
            "day": 7,
            "title": "Reflection & Forward Planning",
            "theme": "Review & Renewal",
            "objectives": [
                "Complete a full written review: energy, sleep, mood, movement, nutrition.",
                "Identify your 3 biggest wins from this week.",
                "Identify 2 habits you want to strengthen in Week 2.",
                "Set one health goal for the next 30 days.",
                "Celebrate — you completed the 7-Day Protocol!",
            ],
            "sleep_guidance": (
                "Compare tonight's sleep to Day 1. Most participants report meaningfully "
                "improved sleep quality by Day 7 — even modest habit changes accumulate. "
                "Commit to maintaining your sleep schedule in Week 2."
            ),
            "nutrition_guidance": (
                "Notice if your relationship with food has shifted — many participants "
                "report reduced cravings, steadier energy, and improved appetite regulation. "
                "These are signs your metabolism is beginning to recalibrate."
            ),
            "movement_guidance": (
                "Active recovery day: a relaxed 25-minute walk or gentle stretch session. "
                "You have built a movement habit this week — honour it."
            ),
            "coach_note": (
                "Seven days doesn't transform your body, but it does transform your awareness. "
                "You now have data about your own patterns, and that knowledge is powerful. "
                "The real work — and the real rewards — begin in Week 2 and beyond."
            ),
        },
    ],
}


def _sanitize(text: str) -> str:
    """Replace non-latin-1 chars with ASCII equivalents for Helvetica compatibility."""
    return (
        text
        .replace("\u2013", "-")   # en-dash
        .replace("\u2014", "--")  # em-dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u2022", "-")   # bullet
        .replace("\u2026", "...") # ellipsis
        .replace("\u00b0", " deg") # degree symbol (latin-1 is \xb0 but some fonts skip)
        .encode("latin-1", errors="replace").decode("latin-1")
    )


class WellnessPDF(FPDF):
    PRIMARY = (46, 139, 87)    # forest green
    ACCENT  = (95, 158, 160)   # cadet blue
    DARK    = (30, 30, 30)
    LIGHT   = (245, 245, 240)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 8, "7-Day Foundational Wellness Protocol", align="C")
        self.ln(2)
        self.set_draw_color(*self.ACCENT)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, text: str, level: int = 1):
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*self.PRIMARY)
        else:
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(*self.ACCENT)
        self.ln(4)
        self.multi_cell(0, 8, _sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        self.multi_cell(0, 6, _sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullet_list(self, items: list[str], bullet: str = "-"):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.DARK)
        for item in items:
            self.set_x(self.l_margin + 4)
            self.multi_cell(
                0, 6,
                _sanitize(f"{bullet}  {item}"),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        self.ln(2)

    def label_block(self, label: str, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 6, _sanitize(label), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.body_text(text)


def generate_pdf(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = WellnessPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)

    # ── Cover page ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*WellnessPDF.PRIMARY)
    pdf.multi_cell(0, 12, PROTOCOL_CONTENT["title"], align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 14)
    pdf.set_text_color(*WellnessPDF.ACCENT)
    pdf.multi_cell(0, 8, PROTOCOL_CONTENT["subtitle"], align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 7, PROTOCOL_CONTENT["overview"], align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── General rules ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("General Rules", level=1)
    pdf.bullet_list(PROTOCOL_CONTENT["general_rules"])

    # ── Dos & Don'ts ──────────────────────────────────────────────────────
    pdf.chapter_title("Dos & Don'ts", level=1)
    pdf.chapter_title("Do", level=2)
    pdf.bullet_list(PROTOCOL_CONTENT["dos_and_donts"]["dos"], bullet="[+]")
    pdf.chapter_title("Don't", level=2)
    pdf.bullet_list(PROTOCOL_CONTENT["dos_and_donts"]["donts"], bullet="[-]")

    # ── Tracking rules ────────────────────────────────────────────────────
    pdf.chapter_title("Tracking Rules", level=1)
    pdf.bullet_list(PROTOCOL_CONTENT["tracking_rules"])

    # ── Day-by-day content ────────────────────────────────────────────────
    for day_data in PROTOCOL_CONTENT["days"]:
        pdf.add_page()
        pdf.chapter_title(
            f"Day {day_data['day']}: {day_data['title']}  —  Theme: {day_data['theme']}",
            level=1,
        )

        pdf.chapter_title("Today's Objectives", level=2)
        pdf.bullet_list(day_data["objectives"])

        pdf.label_block("Sleep Guidance", day_data["sleep_guidance"])
        pdf.label_block("Nutrition Guidance", day_data["nutrition_guidance"])
        pdf.label_block("Movement Guidance", day_data["movement_guidance"])
        pdf.label_block("Coach's Note", day_data["coach_note"])

    pdf.output(str(output_path))
    print(f"✅  Wellness protocol PDF saved to: {output_path}")


if __name__ == "__main__":
    generate_pdf()
