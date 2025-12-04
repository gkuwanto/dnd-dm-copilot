"""Tests for CRD3 data processor."""

from unittest.mock import patch

import pytest

# Import module with numeric prefix using __import__
crd3_module = __import__(
    "dnd_dm_copilot.data.crd3.01_create_training_pairs",
    fromlist=[
        "load_crd3_data",
        "extract_dialogues_from_episode",
        "create_dialogue_context_pairs",
        "create_speaker_interaction_pairs",
        "create_chunk_dialogue_pairs",
        "create_turn_sequence_pairs",
        "create_matt_dm_pairs",
        "create_training_dataset",
        "main",
    ],
)

# Assign imported functions
load_crd3_data = crd3_module.load_crd3_data
extract_dialogues_from_episode = crd3_module.extract_dialogues_from_episode
create_dialogue_context_pairs = crd3_module.create_dialogue_context_pairs
create_speaker_interaction_pairs = crd3_module.create_speaker_interaction_pairs
create_chunk_dialogue_pairs = crd3_module.create_chunk_dialogue_pairs
create_turn_sequence_pairs = crd3_module.create_turn_sequence_pairs
create_matt_dm_pairs = crd3_module.create_matt_dm_pairs
create_training_dataset = crd3_module.create_training_dataset
main = crd3_module.main


@pytest.fixture
def sample_episode_data():
    """Sample episode data matching CRD3 structure."""
    return [
        {
            "CHUNK": "The party enters a tavern and meets the bartender.",
            "TURNS": [
                {
                    "NAMES": ["LIAM"],
                    "UTTERANCES": ["We enter the tavern cautiously."],
                },
                {
                    "NAMES": ["MATT"],
                    "UTTERANCES": [
                        "You see a warm",
                        "inviting space with a roaring fire.",
                    ],
                },
                {
                    "NAMES": ["TRAVIS"],
                    "UTTERANCES": ["I approach the bar."],
                },
            ],
        },
        {
            "CHUNK": "The bartender greets the party.",
            "TURNS": [
                {
                    "NAMES": ["MATT"],
                    "UTTERANCES": ["The bartender nods at you warmly."],
                },
                {
                    "NAMES": ["LAURA", "MARISHA"],
                    "UTTERANCES": ["Hello!", "We need rooms for the night."],
                },
            ],
        },
    ]


@pytest.fixture
def sample_dialogues():
    """Sample dialogues as tuples of (speaker, utterance)."""
    return [
        ("LIAM", "We enter the tavern cautiously."),
        ("MATT", "You see a warm inviting space with a roaring fire."),
        ("TRAVIS", "I approach the bar."),
        ("MATT", "The bartender nods at you warmly."),
        ("LAURA", "Hello! We need rooms for the night."),
    ]


class TestLoadCrd3Data:
    """Tests for load_crd3_data function."""

    def test_load_crd3_data_success(self, temp_dir):
        """Test successful data loading."""
        # Create mock directory structure
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        # Create mock JSON file
        episode_file = c2_dir / "1x01_aligned.json"
        episode_data = [
            {
                "CHUNK": "Test chunk",
                "TURNS": [{"NAMES": ["MATT"], "UTTERANCES": ["Hello"]}],
            }
        ]

        import json

        with open(episode_file, "w") as f:
            json.dump(episode_data, f)

        result = load_crd3_data(str(temp_dir))

        assert len(result) == 1
        assert result[0]["episode_name"] == "1x01"
        assert result[0]["data"] == episode_data

    def test_load_crd3_data_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        result = load_crd3_data(str(temp_dir))

        assert result == []

    def test_load_crd3_data_invalid_json(self, temp_dir, capsys):
        """Test handling of invalid JSON files."""
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        # Create invalid JSON file
        invalid_file = c2_dir / "1x01_aligned.json"
        invalid_file.write_text("not valid json")

        result = load_crd3_data(str(temp_dir))

        # Should return empty list and print error
        assert result == []
        captured = capsys.readouterr()
        assert "Error loading" in captured.out

    def test_load_crd3_data_non_json_files(self, temp_dir):
        """Test that non-JSON files are ignored."""
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        # Create non-JSON file
        (c2_dir / "readme.txt").write_text("Not a JSON file")

        result = load_crd3_data(str(temp_dir))

        assert result == []


class TestExtractDialoguesFromEpisode:
    """Tests for extract_dialogues_from_episode function."""

    def test_extract_dialogues_single_speaker(self):
        """Test extraction with single speaker."""
        episode_data = [
            {
                "CHUNK": "Test",
                "TURNS": [
                    {"NAMES": ["MATT"], "UTTERANCES": ["Hello", "there"]},
                    {"NAMES": ["LIAM"], "UTTERANCES": ["Hi"]},
                ],
            }
        ]

        result = extract_dialogues_from_episode(episode_data)

        assert len(result) == 2
        assert result[0] == ("MATT", "Hello there")
        assert result[1] == ("LIAM", "Hi")

    def test_extract_dialogues_multiple_speakers(self, sample_episode_data):
        """Test extraction with multiple speakers in unison."""
        result = extract_dialogues_from_episode(sample_episode_data)

        # Check that unison speakers are properly formatted
        assert any("in unison" in speaker for speaker, _ in result)

    def test_extract_dialogues_empty_episode(self):
        """Test extraction from empty episode."""
        result = extract_dialogues_from_episode([])

        assert result == []


class TestCreateDialogueContextPairs:
    """Tests for create_dialogue_context_pairs function."""

    def test_create_dialogue_context_pairs_default_window(self, sample_dialogues):
        """Test pair creation with default context window."""
        result = create_dialogue_context_pairs(sample_dialogues)

        assert len(result) > 0
        assert all("query" in pair and "passage" in pair for pair in result)
        assert all("Dialogue context:" in pair["query"] for pair in result)

    def test_create_dialogue_context_pairs_custom_window(self, sample_dialogues):
        """Test pair creation with custom context window."""
        result = create_dialogue_context_pairs(sample_dialogues, context_window=2)

        # With 5 dialogues and window=2, expect 3 pairs
        assert len(result) == 3

    def test_create_dialogue_context_pairs_insufficient_data(self):
        """Test with insufficient dialogues for context window."""
        dialogues = [("MATT", "Hello")]

        result = create_dialogue_context_pairs(dialogues, context_window=3)

        # Not enough dialogues to create pairs
        assert result == []

    def test_create_dialogue_context_pairs_format(self, sample_dialogues):
        """Test that pairs have correct format."""
        result = create_dialogue_context_pairs(sample_dialogues, context_window=2)

        first_pair = result[0]
        assert "LIAM:" in first_pair["query"]
        assert "MATT:" in first_pair["query"]
        assert "TRAVIS:" in first_pair["passage"]


class TestCreateSpeakerInteractionPairs:
    """Tests for create_speaker_interaction_pairs function."""

    def test_create_speaker_interaction_pairs_success(self, sample_dialogues):
        """Test successful speaker interaction pair creation."""
        result = create_speaker_interaction_pairs(sample_dialogues)

        assert len(result) > 0
        assert all("query" in pair and "passage" in pair for pair in result)
        assert all("says:" in pair["query"] for pair in result)
        assert all("responds:" in pair["passage"] for pair in result)

    def test_create_speaker_interaction_pairs_same_speaker(self):
        """Test that pairs are not created for same speaker."""
        dialogues = [
            ("MATT", "First line"),
            ("MATT", "Second line"),
            ("LIAM", "Third line"),
        ]

        result = create_speaker_interaction_pairs(dialogues)

        # Only one pair should be created (MATT -> LIAM)
        assert len(result) == 1
        assert "MATT says:" in result[0]["query"]
        assert "LIAM responds:" in result[0]["passage"]

    def test_create_speaker_interaction_pairs_empty(self):
        """Test with empty dialogues."""
        result = create_speaker_interaction_pairs([])

        assert result == []

    def test_create_speaker_interaction_pairs_single_dialogue(self):
        """Test with single dialogue."""
        result = create_speaker_interaction_pairs([("MATT", "Hello")])

        assert result == []


class TestCreateChunkDialoguePairs:
    """Tests for create_chunk_dialogue_pairs function."""

    def test_create_chunk_dialogue_pairs_success(self, sample_episode_data):
        """Test successful chunk-dialogue pair creation."""
        result = create_chunk_dialogue_pairs(sample_episode_data)

        assert len(result) > 0
        assert all("query" in pair and "passage" in pair for pair in result)
        assert all("Scene context:" in pair["query"] for pair in result)

    def test_create_chunk_dialogue_pairs_multiple_speakers(self):
        """Test chunk pairs with multiple speakers in unison."""
        episode_data = [
            {
                "CHUNK": "Test scene",
                "TURNS": [{"NAMES": ["LAURA", "MARISHA"], "UTTERANCES": ["Together!"]}],
            }
        ]

        result = create_chunk_dialogue_pairs(episode_data)

        assert len(result) == 1
        assert "in unison" in result[0]["passage"]

    def test_create_chunk_dialogue_pairs_empty(self):
        """Test with empty episode data."""
        result = create_chunk_dialogue_pairs([])

        assert result == []

    def test_create_chunk_dialogue_pairs_format(self, sample_episode_data):
        """Test that pairs have correct format."""
        result = create_chunk_dialogue_pairs(sample_episode_data)

        # Check first pair has chunk summary in query
        assert "tavern" in result[0]["query"].lower()


class TestCreateTurnSequencePairs:
    """Tests for create_turn_sequence_pairs function."""

    def test_create_turn_sequence_pairs_default_length(self):
        """Test sequence pair creation with default length."""
        # Need at least 6 dialogues for default sequence_length=5
        dialogues = [
            ("A", "1"),
            ("B", "2"),
            ("C", "3"),
            ("D", "4"),
            ("E", "5"),
            ("F", "6"),
        ]
        result = create_turn_sequence_pairs(dialogues)

        assert len(result) > 0
        assert all("query" in pair and "passage" in pair for pair in result)

    def test_create_turn_sequence_pairs_custom_length(self, sample_dialogues):
        """Test sequence pair creation with custom length."""
        result = create_turn_sequence_pairs(sample_dialogues, sequence_length=3)

        # With 5 dialogues and length=3, expect 2 pairs
        assert len(result) == 2

    def test_create_turn_sequence_pairs_insufficient_data(self):
        """Test with insufficient dialogues for sequence."""
        dialogues = [("MATT", "Hello"), ("LIAM", "Hi")]

        result = create_turn_sequence_pairs(dialogues, sequence_length=5)

        assert result == []

    def test_create_turn_sequence_pairs_split(self):
        """Test that sequences are properly split."""
        dialogues = [
            ("A", "1"),
            ("B", "2"),
            ("C", "3"),
            ("D", "4"),
            ("E", "5"),
            ("F", "6"),
        ]

        result = create_turn_sequence_pairs(dialogues, sequence_length=4)

        # First pair should split [A,B,C,D] into query=[A,B] passage=[C,D]
        first_pair = result[0]
        assert "A:" in first_pair["query"]
        assert "B:" in first_pair["query"]
        assert "C:" in first_pair["passage"]
        assert "D:" in first_pair["passage"]


class TestCreateMattDmPairs:
    """Tests for create_matt_dm_pairs function."""

    def test_create_matt_dm_pairs_success(self, sample_dialogues):
        """Test successful DM pair creation."""
        result = create_matt_dm_pairs(sample_dialogues)

        assert len(result) > 0
        assert all("query" in pair and "passage" in pair for pair in result)
        assert all("Player says:" in pair["query"] for pair in result)
        assert all("DM responds:" in pair["passage"] for pair in result)

    def test_create_matt_dm_pairs_only_matt_responses(self):
        """Test that only Matt's responses are captured."""
        dialogues = [
            ("LIAM", "I attack"),
            ("MATT", "Roll for attack"),
            ("TRAVIS", "I hide"),
            ("LIAM", "I search"),
        ]

        result = create_matt_dm_pairs(dialogues)

        # Only one pair: LIAM -> MATT
        assert len(result) == 1
        assert "I attack" in result[0]["query"]
        assert "Roll for attack" in result[0]["passage"]

    def test_create_matt_dm_pairs_no_matt(self):
        """Test with no Matt dialogues."""
        dialogues = [
            ("LIAM", "Hello"),
            ("TRAVIS", "Hi"),
            ("LAURA", "Hey"),
        ]

        result = create_matt_dm_pairs(dialogues)

        assert result == []

    def test_create_matt_dm_pairs_consecutive_matt(self):
        """Test that consecutive Matt dialogues create pairs."""
        dialogues = [
            ("MATT", "First"),
            ("MATT", "Second"),
            ("LIAM", "Player line"),
        ]

        result = create_matt_dm_pairs(dialogues)

        # One pair should be created (MATT -> MATT)
        assert len(result) == 1
        assert "First" in result[0]["query"]
        assert "Second" in result[0]["passage"]

    def test_create_matt_dm_pairs_empty(self):
        """Test with empty dialogues."""
        result = create_matt_dm_pairs([])

        assert result == []


class TestCreateTrainingDataset:
    """Tests for create_training_dataset function."""

    def test_create_training_dataset_all_methods(self, sample_episode_data, capsys):
        """Test dataset creation with all methods."""
        episodes = [{"episode_name": "test_ep", "data": sample_episode_data}]

        result = create_training_dataset(episodes)

        assert len(result) > 0
        # Check that various methods contributed
        captured = capsys.readouterr()
        assert "dialogue context pairs" in captured.out
        assert "speaker interaction pairs" in captured.out

    def test_create_training_dataset_single_method(self, sample_episode_data):
        """Test dataset creation with single method."""
        episodes = [{"episode_name": "test_ep", "data": sample_episode_data}]

        result = create_training_dataset(episodes, methods=["dialogue_context"])

        assert len(result) > 0
        assert all("Dialogue context:" in pair["query"] for pair in result)

    def test_create_training_dataset_multiple_episodes(self, sample_episode_data):
        """Test dataset creation with multiple episodes."""
        episodes = [
            {"episode_name": "ep1", "data": sample_episode_data},
            {"episode_name": "ep2", "data": sample_episode_data},
        ]

        result = create_training_dataset(episodes, methods=["matt_dm"])

        # Should have pairs from both episodes
        assert len(result) > 0

    def test_create_training_dataset_empty_methods(self, sample_episode_data):
        """Test with empty methods list."""
        episodes = [{"episode_name": "test_ep", "data": sample_episode_data}]

        result = create_training_dataset(episodes, methods=[])

        assert result == []

    def test_create_training_dataset_empty_episodes(self):
        """Test with empty episodes list."""
        result = create_training_dataset([])

        assert result == []

    def test_create_training_dataset_invalid_method(self, sample_episode_data):
        """Test with invalid method name."""
        episodes = [{"episode_name": "test_ep", "data": sample_episode_data}]

        result = create_training_dataset(episodes, methods=["invalid_method"])

        # Should return empty list (no valid methods)
        assert result == []


class TestMain:
    """Tests for main function."""

    def test_main_success(self, temp_dir):
        """Test successful main execution."""
        # Create mock directory structure
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        # Create mock episode file
        episode_file = c2_dir / "1x01_aligned.json"
        episode_data = [
            {
                "CHUNK": "Test",
                "TURNS": [
                    {"NAMES": ["LIAM"], "UTTERANCES": ["Hello"]},
                    {"NAMES": ["MATT"], "UTTERANCES": ["Hi"]},
                ],
            }
        ]

        import json

        with open(episode_file, "w") as f:
            json.dump(episode_data, f)

        with (
            patch("os.getcwd", return_value=str(temp_dir)),
            patch("huggingface_hub.create_repo") as mock_create,
            patch("huggingface_hub.upload_file") as mock_upload,
            patch.object(crd3_module, "HF_TOKEN", "test-token"),
        ):
            main()

            mock_create.assert_called_once()
            mock_upload.assert_called_once()

    def test_main_no_episodes(self, temp_dir, capsys):
        """Test main with no episodes found."""
        # Create empty directory structure
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        with (
            patch("os.getcwd", return_value=str(temp_dir)),
            patch("huggingface_hub.create_repo"),
            patch("huggingface_hub.upload_file"),
        ):
            main()

            captured = capsys.readouterr()
            assert "Loaded 0 episodes" in captured.out

    def test_main_upload_failure(self, temp_dir, capsys):
        """Test main handles upload failure gracefully."""
        c2_dir = temp_dir / "data" / "aligned data" / "c=2"
        c2_dir.mkdir(parents=True)

        episode_file = c2_dir / "1x01_aligned.json"
        episode_data = [
            {
                "CHUNK": "Test",
                "TURNS": [
                    {"NAMES": ["LIAM"], "UTTERANCES": ["Hello"]},
                    {"NAMES": ["MATT"], "UTTERANCES": ["Hi"]},
                ],
            }
        ]

        import json

        with open(episode_file, "w") as f:
            json.dump(episode_data, f)

        with (
            patch("os.getcwd", return_value=str(temp_dir)),
            patch("huggingface_hub.create_repo"),
            patch(
                "huggingface_hub.upload_file", side_effect=Exception("Upload failed")
            ),
            patch.object(crd3_module, "HF_TOKEN", "test-token"),
            pytest.raises(Exception, match="Upload failed"),
        ):
            main()
