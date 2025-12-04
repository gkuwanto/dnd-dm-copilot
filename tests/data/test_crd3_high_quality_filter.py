"""Tests for CRD3 high_quality_filter processor."""

from unittest.mock import patch

import pytest

# Import module with numeric prefix using __import__
hq_filter_module = __import__(
    "dnd_dm_copilot.data.crd3.04_high_quality_filter",
    fromlist=[
        "has_meaningful_content",
        "has_dm_narration",
        "has_rules_content",
        "is_conversational_filler",
        "calculate_quality_score",
        "filter_high_quality_pairs",
        "print_quality_analysis",
        "main",
    ],
)

# Assign imported functions
has_meaningful_content = hq_filter_module.has_meaningful_content
has_dm_narration = hq_filter_module.has_dm_narration
has_rules_content = hq_filter_module.has_rules_content
is_conversational_filler = hq_filter_module.is_conversational_filler
calculate_quality_score = hq_filter_module.calculate_quality_score
filter_high_quality_pairs = hq_filter_module.filter_high_quality_pairs
print_quality_analysis = hq_filter_module.print_quality_analysis
main = hq_filter_module.main


class TestHasMeaningfulContent:
    """Tests for has_meaningful_content function."""

    def test_meaningful_content_long_text(self):
        """Test detection of meaningful long text."""
        text = "DM responds: You see a large chamber with stone walls and flickering torches"
        assert has_meaningful_content(text) is True

    def test_meaningful_content_short_text(self):
        """Test rejection of short text."""
        text = "DM responds: You see"
        assert has_meaningful_content(text) is False

    def test_low_quality_simple_acknowledgment(self):
        """Test rejection of simple acknowledgments."""
        assert has_meaningful_content("DM responds: Okay.") is False
        assert has_meaningful_content("DM responds: Yes.") is False
        assert has_meaningful_content("DM responds: Sure.") is False

    def test_low_quality_just_numbers(self):
        """Test rejection of just numbers."""
        assert has_meaningful_content("DM responds: 15.") is False

    def test_low_quality_incomplete_statement(self):
        """Test rejection of incomplete statements."""
        assert has_meaningful_content("DM responds: You can.") is False
        assert has_meaningful_content("DM responds: You do.") is False

    def test_meaningful_substantial_response(self):
        """Test acceptance of substantial responses."""
        text = "DM responds: You enter a dark cavern with the sound of dripping water echoing"
        assert has_meaningful_content(text) is True


class TestHasDmNarration:
    """Tests for has_dm_narration function."""

    def test_narration_sensory_description(self):
        """Test detection of sensory descriptions."""
        assert has_dm_narration("You see a dragon in front of you") is True
        assert has_dm_narration("You hear a distant rumble") is True
        assert has_dm_narration("You notice something strange") is True
        assert has_dm_narration("You find a hidden door") is True

    def test_narration_environmental_description(self):
        """Test detection of environmental descriptions."""
        assert has_dm_narration("The room is filled with darkness") is True
        assert has_dm_narration("The air feels heavy and oppressive") is True
        assert has_dm_narration("A shadow moves across the wall") is True

    def test_narration_entities(self):
        """Test detection of entity descriptions."""
        assert has_dm_narration("A creature lurks in the corner") is True
        assert has_dm_narration("The monster roars at you") is True
        assert has_dm_narration("You see a strange figure") is True

    def test_narration_locations(self):
        """Test detection of location descriptions."""
        assert has_dm_narration("The door creaks open") is True
        assert has_dm_narration("You enter a long hallway") is True
        assert has_dm_narration("The chamber is vast") is True

    def test_no_narration(self):
        """Test text without narration indicators."""
        assert has_dm_narration("Roll for initiative") is False
        assert has_dm_narration("What do you want to do?") is False

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert has_dm_narration("YOU SEE a dragon") is True
        assert has_dm_narration("The DARKNESS surrounds you") is True


class TestHasRulesContent:
    """Tests for has_rules_content function."""

    def test_rules_dice_rolling(self):
        """Test detection of dice rolling content."""
        assert has_rules_content("Roll a d20 for initiative") is True
        assert has_rules_content("Make a saving throw") is True
        assert has_rules_content("Roll for attack") is True

    def test_rules_combat_mechanics(self):
        """Test detection of combat mechanics."""
        assert has_rules_content("You take 10 damage") is True
        assert has_rules_content("The attack hits your AC") is True
        assert has_rules_content("Your hit points are reduced") is True

    def test_rules_spell_casting(self):
        """Test detection of spell casting."""
        assert has_rules_content("You cast fireball") is True
        assert has_rules_content("Concentration check required") is True
        assert has_rules_content("Use a spell slot") is True

    def test_rules_advantage_disadvantage(self):
        """Test detection of advantage/disadvantage."""
        assert has_rules_content("You have advantage on this roll") is True
        assert has_rules_content("Roll with disadvantage") is True

    def test_rules_action_economy(self):
        """Test detection of action economy."""
        assert has_rules_content("Use your action to attack") is True
        assert has_rules_content("That would be a bonus action") is True
        assert has_rules_content("You can use your reaction") is True

    def test_rules_class_resources(self):
        """Test detection of class resources."""
        assert has_rules_content("Spend a ki point") is True
        assert has_rules_content("Enter a rage") is True
        assert has_rules_content("Use your proficiency bonus") is True

    def test_no_rules_content(self):
        """Test text without rules content."""
        assert has_rules_content("You see a beautiful sunset") is False
        assert has_rules_content("The tavern is warm") is False

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert has_rules_content("ROLL a D20") is True
        assert has_rules_content("Take DAMAGE") is True


class TestIsConversationalFiller:
    """Tests for is_conversational_filler function."""

    def test_short_query_is_filler(self):
        """Test that short queries are considered filler."""
        assert is_conversational_filler("Player says: Hi") is True
        assert is_conversational_filler("Player says: What") is True

    def test_meta_questions_are_filler(self):
        """Test detection of meta-conversation."""
        assert is_conversational_filler("can i do this") is True
        assert is_conversational_filler("do i get advantage") is True
        assert is_conversational_filler("how do I attack") is True
        assert is_conversational_filler("what is my AC") is True

    def test_simple_acknowledgments_are_filler(self):
        """Test detection of simple acknowledgments."""
        assert is_conversational_filler("okay") is True
        assert is_conversational_filler("yes") is True
        assert is_conversational_filler("sure") is True

    def test_substantial_query_not_filler(self):
        """Test that substantial queries are not filler."""
        assert (
            is_conversational_filler(
                "Player says: I want to carefully search the room for traps"
            )
            is False
        )
        assert (
            is_conversational_filler(
                "Player says: I approach the dragon and attempt to negotiate"
            )
            is False
        )

    def test_with_speaker_prefix(self):
        """Test filler detection with speaker prefix."""
        assert is_conversational_filler("Player says: okay") is True
        assert is_conversational_filler("Player says: what") is True


class TestCalculateQualityScore:
    """Tests for calculate_quality_score function."""

    def test_high_quality_pair(self):
        """Test scoring of high-quality pair."""
        pair = {
            "query": "Player says: I want to carefully search the ancient library for clues about the artifact",
            "passage": "DM responds: You see ancient tomes lining the walls as you make a perception check with advantage because of your careful approach",
        }

        score = calculate_quality_score(pair)

        # Should have high score: meaningful + narration + rules + length bonuses
        assert score >= 4.0

    def test_low_quality_pair(self):
        """Test scoring of low-quality pair."""
        pair = {"query": "Player says: Hi", "passage": "DM responds: Yes okay"}

        score = calculate_quality_score(pair)

        # Should have low score
        assert score < 3.0

    def test_narration_bonus(self):
        """Test that narration increases score."""
        pair_with_narration = {
            "query": "Player says: I look around the room",
            "passage": "DM responds: You see flickering torches casting shadows on ancient stone walls",
        }

        pair_without_narration = {
            "query": "Player says: I look around the room",
            "passage": "DM responds: Roll a perception check",
        }

        score_with = calculate_quality_score(pair_with_narration)
        score_without = calculate_quality_score(pair_without_narration)

        assert score_with > score_without

    def test_rules_content_bonus(self):
        """Test that rules content increases score."""
        pair_with_rules = {
            "query": "Player says: I attack the goblin",
            "passage": "DM responds: Roll a d20 and add your attack modifier",
        }

        pair_without_rules = {
            "query": "Player says: I attack the goblin",
            "passage": "DM responds: You swing your sword",
        }

        score_with = calculate_quality_score(pair_with_rules)
        score_without = calculate_quality_score(pair_without_rules)

        assert score_with > score_without

    def test_length_bonus(self):
        """Test that longer passages get bonus."""
        pair_long = {
            "query": "Player says: I search the room",
            "passage": "DM responds: "
            + " ".join(["word"] * 35),  # 35 words
        }

        pair_short = {
            "query": "Player says: I search the room",
            "passage": "DM responds: " + " ".join(["word"] * 15),  # 15 words
        }

        score_long = calculate_quality_score(pair_long)
        score_short = calculate_quality_score(pair_short)

        assert score_long > score_short


class TestFilterHighQualityPairs:
    """Tests for filter_high_quality_pairs function."""

    def test_filter_high_quality_pairs(self, capsys):
        """Test filtering for high-quality pairs."""
        pairs = [
            {
                "query": "Player says: I carefully search the ancient ruins",
                "passage": "DM responds: You see crumbling walls and hear the wind whistling through broken windows",
            },
            {"query": "Hi", "passage": "Yes"},
            {
                "query": "Player says: I attack with my sword",
                "passage": "DM responds: Roll a d20 and add your strength modifier",
            },
        ]

        result = filter_high_quality_pairs(pairs, target_count=2, min_quality_score=3.0)

        # Should filter out the low-quality pair
        assert len(result) <= 2

        captured = capsys.readouterr()
        assert "Calculating quality scores" in captured.out

    def test_filter_respects_target_count(self, capsys):
        """Test that filter respects target count."""
        # Create many high-quality pairs
        pairs = [
            {
                "query": "Player says: I search the room for clues",
                "passage": "DM responds: You see ancient symbols carved into the walls as you roll perception",
            }
        ] * 100

        result = filter_high_quality_pairs(pairs, target_count=50)

        assert len(result) == 50

    def test_filter_empty_pairs(self, capsys):
        """Test filtering with empty pairs."""
        result = filter_high_quality_pairs([], target_count=10)

        assert result == []

    def test_filter_no_pairs_meet_threshold(self, capsys):
        """Test when no pairs meet quality threshold."""
        pairs = [{"query": "Hi", "passage": "Yes"}] * 10

        result = filter_high_quality_pairs(pairs, min_quality_score=5.0)

        assert len(result) == 0


class TestPrintQualityAnalysis:
    """Tests for print_quality_analysis function."""

    def test_print_quality_analysis(self, capsys):
        """Test quality analysis printing."""
        pairs = [
            {
                "query": "Player says: I search",
                "passage": "DM responds: You see a room",
            },
            {
                "query": "Player says: I attack",
                "passage": "DM responds: Roll a d20",
            },
        ]

        print_quality_analysis(pairs)

        captured = capsys.readouterr()
        assert "Quality Analysis:" in captured.out
        assert "Pairs with DM narration:" in captured.out
        assert "Pairs with rules content:" in captured.out
        assert "Average query length:" in captured.out
        assert "Average passage length:" in captured.out


class TestMain:
    """Tests for main function."""

    def test_main_success(self, temp_dir, capsys):
        """Test successful main execution."""
        import json

        input_file = temp_dir / "crd3_training_pairs_dm_only.json"
        output_file = temp_dir / "crd3_training_pairs_high_quality.json"

        pairs = [
            {
                "query": "Player says: I want to search the ancient library carefully",
                "passage": "DM responds: You see dusty tomes and scrolls as you roll perception with advantage",
            }
        ] * 100

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        with (
            patch("builtins.open", side_effect=[open(input_file, "r"), open(output_file, "w")]),
            patch("json.load", return_value=pairs),
            patch("json.dump") as mock_dump,
        ):
            main()

            assert mock_dump.called

    def test_main_full_execution(self, temp_dir, capsys):
        """Test full main execution."""
        import json

        input_file = temp_dir / "crd3_training_pairs_dm_only.json"
        output_file = temp_dir / "crd3_training_pairs_high_quality.json"

        # Create pairs with varying quality
        pairs = [
            {
                "query": "Player says: I carefully examine the ancient ruins for any signs of the artifact",
                "passage": "DM responds: You see crumbling stone walls with mystical runes as you make an investigation check",
            }
        ] * 50
        pairs += [{"query": "Hi", "passage": "Yes"}] * 50

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        with (
            patch(
                "builtins.open",
                side_effect=[
                    open(input_file, "r", encoding="utf-8"),
                    open(output_file, "w", encoding="utf-8"),
                ],
            ),
        ):
            main()

        captured = capsys.readouterr()
        assert "Loaded 100 DM response pairs" in captured.out
        assert "Selected" in captured.out
        assert "Quality Analysis:" in captured.out
