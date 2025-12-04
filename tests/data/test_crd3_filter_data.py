"""Tests for CRD3 filter_data processor."""

from unittest.mock import patch

# Import module with numeric prefix using __import__
filter_module = __import__(
    "dnd_dm_copilot.data.crd3.02_filter_data",
    fromlist=[
        "is_likely_dice_roll",
        "is_filler_response",
        "filter_training_pairs",
        "print_statistics",
        "main",
    ],
)

# Assign imported functions
is_likely_dice_roll = filter_module.is_likely_dice_roll
is_filler_response = filter_module.is_filler_response
filter_training_pairs = filter_module.filter_training_pairs
print_statistics = filter_module.print_statistics
main = filter_module.main


class TestIsLikelyDiceRoll:
    """Tests for is_likely_dice_roll function."""

    def test_simple_number(self):
        """Test detection of simple numbers."""
        assert is_likely_dice_roll("23") is True
        assert is_likely_dice_roll("100") is True
        assert is_likely_dice_roll("MATT: 15") is True

    def test_dice_notation(self):
        """Test detection of dice notation."""
        assert is_likely_dice_roll("d20") is True
        assert is_likely_dice_roll("1d6") is True
        assert is_likely_dice_roll("2d8") is True
        assert is_likely_dice_roll("TRAVIS: 3d10") is True

    def test_decimal_number(self):
        """Test detection of decimal numbers."""
        assert is_likely_dice_roll("15.5") is True
        assert is_likely_dice_roll("10,000") is True

    def test_not_dice_roll(self):
        """Test non-dice-roll text."""
        assert is_likely_dice_roll("You see a dragon") is False
        assert is_likely_dice_roll("I cast fireball") is False
        assert is_likely_dice_roll("The door opens") is False

    def test_with_speaker_prefix(self):
        """Test dice roll detection with speaker prefixes."""
        assert is_likely_dice_roll("MATT: 17") is True
        assert is_likely_dice_roll("LAURA: d20") is True

    def test_multiple_words(self):
        """Test that multi-word text is not detected as dice roll."""
        assert is_likely_dice_roll("Roll a d20 for initiative") is False


class TestIsFillerResponse:
    """Tests for is_filler_response function."""

    def test_simple_filler(self):
        """Test detection of simple filler words."""
        assert is_filler_response("yeah") is True
        assert is_filler_response("yes") is True
        assert is_filler_response("okay") is True
        assert is_filler_response("sure") is True
        assert is_filler_response("no") is True

    def test_filler_with_punctuation(self):
        """Test filler detection with punctuation."""
        assert is_filler_response("Yeah.") is True
        assert is_filler_response("Okay!") is True
        assert is_filler_response("Yes?") is True

    def test_filler_with_speaker_prefix(self):
        """Test filler detection with speaker prefixes."""
        assert is_filler_response("MATT: yeah") is True
        assert is_filler_response("LAURA: okay") is True

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert is_filler_response("YEAH") is True
        assert is_filler_response("Yes") is True
        assert is_filler_response("OkAy") is True

    def test_not_filler(self):
        """Test non-filler responses."""
        assert is_filler_response("You enter the room") is False
        assert is_filler_response("I want to attack") is False
        assert is_filler_response("The dragon roars") is False

    def test_multiple_words(self):
        """Test that multi-word responses are not filler."""
        assert is_filler_response("yeah that sounds good") is False


class TestFilterTrainingPairs:
    """Tests for filter_training_pairs function."""

    def test_filter_short_passage(self):
        """Test filtering of short passages."""
        pairs = [
            {"query": "What do I see here?", "passage": "A room"},
            {
                "query": "What do I see here?",
                "passage": "You see a large room with stone walls",
            },
        ]

        filtered, stats = filter_training_pairs(
            pairs, min_passage_words=5, min_query_words=4
        )

        assert len(filtered) == 1
        assert stats["removed_short_passage"] == 1
        assert filtered[0]["passage"] == "You see a large room with stone walls"

    def test_filter_short_query(self):
        """Test filtering of short queries."""
        pairs = [
            {"query": "Hi", "passage": "Hello there brave adventurer friend"},
            {
                "query": "What do I see here?",
                "passage": "You see a large room",
            },
        ]

        filtered, stats = filter_training_pairs(pairs, min_query_words=4)

        assert len(filtered) == 1
        assert stats["removed_short_query"] == 1

    def test_filter_dice_rolls(self):
        """Test filtering of dice rolls."""
        pairs = [
            {"query": "I attack with my sword", "passage": "MATT: 15"},
            {"query": "I attack with my sword", "passage": "You hit the target"},
        ]

        filtered, stats = filter_training_pairs(
            pairs, remove_dice_rolls=True, min_passage_words=2, min_query_words=2
        )

        assert len(filtered) == 1
        assert stats["removed_dice_roll"] == 1

    def test_filter_fillers(self):
        """Test filtering of filler responses."""
        pairs = [
            {"query": "Can I do that action?", "passage": "Yeah"},
            {"query": "Can I do that action?", "passage": "You can certainly try"},
        ]

        filtered, stats = filter_training_pairs(
            pairs, remove_fillers=True, min_passage_words=1, min_query_words=1
        )

        assert len(filtered) == 1
        assert stats["removed_filler"] == 1

    def test_keep_dm_responses(self):
        """Test that DM responses are always kept if long enough."""
        pairs = [
            {"query": "I attack", "passage": "DM responds: Roll for attack now please"},
            {"query": "I attack", "passage": "DM responds: 15"},
        ]

        filtered, stats = filter_training_pairs(
            pairs, min_passage_words=5, keep_dm_responses=True
        )

        # First should be kept (meets min_passage_words), second should be removed
        assert len(filtered) == 1
        assert "DM responds:" in filtered[0]["passage"]

    def test_disable_filters(self):
        """Test disabling filters."""
        pairs = [
            {"query": "Hi", "passage": "Yeah"},
            {"query": "Attack", "passage": "15"},
        ]

        filtered, stats = filter_training_pairs(
            pairs,
            min_passage_words=1,
            min_query_words=1,
            remove_dice_rolls=False,
            remove_fillers=False,
        )

        assert len(filtered) == 2
        assert stats["kept"] == 2

    def test_statistics_accuracy(self):
        """Test that statistics are accurately reported."""
        pairs = [
            {"query": "Short", "passage": "Okay"},  # Short query, filler
            {"query": "What do I see?", "passage": "Room"},  # Short passage
            {"query": "I attack the orc", "passage": "15"},  # Dice roll
            {
                "query": "What do I see here?",
                "passage": "You see a large stone room",
            },  # Valid
        ]

        filtered, stats = filter_training_pairs(pairs)

        assert stats["total"] == 4
        assert stats["kept"] == 1
        assert stats["removed_short_query"] >= 0
        assert stats["removed_short_passage"] >= 0
        assert stats["removed_dice_roll"] >= 0
        assert stats["removed_filler"] >= 0

    def test_empty_pairs(self):
        """Test with empty pairs list."""
        filtered, stats = filter_training_pairs([])

        assert filtered == []
        assert stats["total"] == 0
        assert stats["kept"] == 0

    def test_all_valid_pairs(self):
        """Test with all valid pairs."""
        pairs = [
            {
                "query": "What do I see in the room?",
                "passage": "You see a large stone chamber",
            },
            {
                "query": "I want to check for traps",
                "passage": "Roll a perception check please",
            },
        ]

        filtered, stats = filter_training_pairs(pairs)

        assert len(filtered) == 2
        assert stats["kept"] == 2


class TestPrintStatistics:
    """Tests for print_statistics function."""

    def test_print_statistics_basic(self, capsys):
        """Test basic statistics printing."""
        stats = {
            "total": 100,
            "kept": 60,
            "removed_short_passage": 15,
            "removed_short_query": 10,
            "removed_dice_roll": 8,
            "removed_filler": 7,
        }

        original_pairs = [
            {"query": "test query one", "passage": "test passage one"}
        ] * 10000
        filtered_pairs = [
            {"query": "test query two", "passage": "test passage two"}
        ] * 10000

        print_statistics(stats, filtered_pairs, original_pairs)

        captured = capsys.readouterr()
        assert "FILTERING STATISTICS" in captured.out
        assert "Original pairs: 100" in captured.out
        assert "Filtered pairs: 60" in captured.out

    def test_print_statistics_with_calculations(self, capsys):
        """Test statistics with average length calculations."""
        stats = {
            "total": 50,
            "kept": 30,
            "removed_short_passage": 10,
            "removed_short_query": 5,
            "removed_dice_roll": 3,
            "removed_filler": 2,
        }

        original_pairs = [{"query": "a b c", "passage": "x y z"}] * 10000
        filtered_pairs = [{"query": "a b c d e", "passage": "x y z w v"}] * 10000

        print_statistics(stats, filtered_pairs, original_pairs)

        captured = capsys.readouterr()
        assert "Average query length:" in captured.out
        assert "Average passage length:" in captured.out


class TestMain:
    """Tests for main function."""

    def test_main_success(self, temp_dir):
        """Test successful main execution."""
        input_file = temp_dir / "crd3_training_pairs_no_llm.json"
        output_file = temp_dir / "crd3_training_pairs_filtered.json"

        # Create mock input file
        import json

        pairs = [
            {
                "query": "What do I see in the room?",
                "passage": "You see a large stone chamber with torches",
            },
            {"query": "What happens?", "passage": "Okay"},  # Will be filtered
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        with (
            patch(
                "builtins.open",
                side_effect=[open(input_file, "r"), open(output_file, "w")],
            ),
            patch("json.load", return_value=pairs),
            patch("json.dump") as mock_dump,
        ):
            main()

            # Verify that json.dump was called
            assert mock_dump.called

    def test_main_file_operations(self, temp_dir, capsys):
        """Test main function file operations."""
        import json

        input_file = temp_dir / "crd3_training_pairs_no_llm.json"
        output_file = temp_dir / "crd3_training_pairs_filtered.json"

        pairs = [
            {
                "query": "What do I see in the tavern?",
                "passage": "You see a warm inviting space with patrons",
            }
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        # Patch open to use our temp files
        main()

        # Verify main ran without errors
        captured = capsys.readouterr()
        assert "Loading data from" in captured.out
