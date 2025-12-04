"""Tests for CRD3 extract_dm_only processor."""

from unittest.mock import patch

# Import module with numeric prefix using __import__
extract_dm_module = __import__(
    "dnd_dm_copilot.data.crd3.03_extract_dm_only",
    fromlist=["extract_dm_pairs", "main"],
)

# Assign imported functions
extract_dm_pairs = extract_dm_module.extract_dm_pairs
main = extract_dm_module.main


class TestExtractDmPairs:
    """Tests for extract_dm_pairs function."""

    def test_extract_dm_pairs_success(self, temp_dir, capsys):
        """Test successful DM pair extraction."""
        import json

        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.json"

        pairs = [
            {
                "query": "Player says: I attack",
                "passage": "DM responds: Roll for attack",
            },
            {"query": "What do I see?", "passage": "You see a room"},
            {
                "query": "Player says: I search",
                "passage": "DM responds: Roll perception",
            },
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        extract_dm_pairs(str(input_file), str(output_file))

        # Verify output file contains only DM pairs
        with open(output_file, "r") as f:
            result = json.load(f)

        assert len(result) == 2
        assert all("DM responds:" in p["passage"] for p in result)

        # Check console output
        captured = capsys.readouterr()
        assert "Extracted 2 DM response pairs" in captured.out

    def test_extract_dm_pairs_no_dm_pairs(self, temp_dir, capsys):
        """Test extraction when no DM pairs exist."""
        import json

        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.json"

        pairs = [
            {"query": "What do I see?", "passage": "You see a room"},
            {"query": "I search", "passage": "You find nothing"},
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        extract_dm_pairs(str(input_file), str(output_file))

        with open(output_file, "r") as f:
            result = json.load(f)

        assert len(result) == 0

        captured = capsys.readouterr()
        assert "Extracted 0 DM response pairs" in captured.out

    def test_extract_dm_pairs_all_dm_pairs(self, temp_dir, capsys):
        """Test extraction when all pairs are DM pairs."""
        import json

        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.json"

        pairs = [
            {
                "query": "Player says: action one",
                "passage": "DM responds: response one",
            },
            {
                "query": "Player says: action two",
                "passage": "DM responds: response two",
            },
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        extract_dm_pairs(str(input_file), str(output_file))

        with open(output_file, "r") as f:
            result = json.load(f)

        assert len(result) == 2
        assert all("DM responds:" in p["passage"] for p in result)

        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_extract_dm_pairs_statistics(self, temp_dir, capsys):
        """Test that statistics are correctly calculated."""
        import json

        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.json"

        # Create pairs with varying word counts
        pairs = [
            {
                "query": "Player says: I attack the goblin with my sword",
                "passage": (
                    "DM responds: Roll an attack roll using your strength modifier"
                ),
            },
            {"query": "What?", "passage": "Nothing special"},
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        extract_dm_pairs(str(input_file), str(output_file))

        captured = capsys.readouterr()
        assert "DM Response Pair Statistics:" in captured.out
        assert "Average query length:" in captured.out
        assert "Average passage length:" in captured.out

    def test_extract_dm_pairs_examples(self, temp_dir, capsys):
        """Test that examples are printed."""
        import json

        input_file = temp_dir / "input.json"
        output_file = temp_dir / "output.json"

        pairs = [
            {
                "query": "Player says: test query",
                "passage": "DM responds: test response",
            }
        ] * 5

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        extract_dm_pairs(str(input_file), str(output_file))

        captured = capsys.readouterr()
        assert "Example DM Response Pairs:" in captured.out
        assert "Example 1" in captured.out


class TestMain:
    """Tests for main function."""

    def test_main_success(self, temp_dir, capsys):
        """Test successful main execution."""
        import json

        input_file = temp_dir / "crd3_training_pairs_filtered.json"
        output_file = temp_dir / "crd3_training_pairs_dm_only.json"

        pairs = [
            {
                "query": "Player says: I attack",
                "passage": "DM responds: Roll for attack",
            },
            {"query": "What?", "passage": "Nothing"},
        ]

        with open(input_file, "w") as f:
            json.dump(pairs, f)

        # Patch the file paths in main
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                side_effect=[open(input_file, "r"), open(output_file, "w")],
            ),
            patch("json.load", return_value=pairs),
            patch("json.dump") as mock_dump,
        ):
            main()

            # Verify dump was called
            assert mock_dump.called

    def test_main_file_not_found(self, capsys):
        """Test main when input file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            main()

            captured = capsys.readouterr()
            assert "Error:" in captured.out
            assert "not found" in captured.out

    def test_main_full_execution(self, temp_dir, capsys):
        """Test full main execution with real files."""
        import json
        import os

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            input_file = temp_dir / "crd3_training_pairs_filtered.json"
            output_file = temp_dir / "crd3_training_pairs_dm_only.json"

            pairs = [
                {
                    "query": "Player says: I want to attack the dragon",
                    "passage": "DM responds: Roll an attack roll please",
                },
                {"query": "What do I see?", "passage": "A big room"},
                {
                    "query": "Player says: I cast fireball",
                    "passage": "DM responds: The enemies take damage",
                },
            ]

            with open(input_file, "w") as f:
                json.dump(pairs, f)

            # Run main (it will look for files in current directory)
            with (
                patch("os.path.exists", return_value=True),
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
            assert "Extracted 2 DM response pairs" in captured.out

        finally:
            os.chdir(original_cwd)
