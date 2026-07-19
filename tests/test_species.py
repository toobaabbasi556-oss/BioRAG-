from biorag.ingest.species import is_animal_study


def test_flags_canine_study():
    assert is_animal_study(
        "Acoustic pressure wave therapy in dogs with joint disease",
        "Twelve dogs were treated.",
    )


def test_flags_equine_study():
    assert is_animal_study(
        "Equine upper limb kinematics during trot",
        "Six horses were recorded.",
    )


def test_flags_murine_study():
    assert is_animal_study("Hyperlocomotion in mice", "Mice received the drug.")


def test_allows_human_gait_study():
    assert not is_animal_study(
        "Gait analysis in patients with Parkinson's disease",
        "Thirty participants walked over a force plate.",
    )


def test_word_boundary_avoids_false_positive_on_ratio():
    """'rat' must not fire inside 'ratio' — a common word in biomechanics."""
    assert not is_animal_study(
        "Stride length to cadence ratio in runners",
        "We computed the ratio for each participant.",
    )


def test_word_boundary_avoids_false_positive_on_pigment():
    assert not is_animal_study(
        "Skin pigmentation and marker tracking accuracy",
        "Pigment density affected optical tracking.",
    )


def test_late_mention_does_not_flag():
    """A human study citing animal work in its discussion is still human."""
    text = "Human participants walked on a treadmill. " + ("Filler text. " * 60)
    assert not is_animal_study("Human treadmill gait", text + "Compare with rat models.")


def test_plug_in_gait_is_not_an_animal_study():
    """'PiG' is the Plug-in Gait model — a human gait standard, not a pig."""
    assert not is_animal_study(
        "Comparison of Plug-in Gait and CGM2.3 models for joint kinematics",
        "Twenty healthy adults were assessed with the Plug-in Gait marker set.",
    )


def test_human_calf_muscle_is_not_an_animal_study():
    assert not is_animal_study(
        "Lower limb muscle activation in response to calf muscle EMG biofeedback",
        "Participants received biofeedback targeting the calf muscles.",
    )


def test_porcine_still_flagged_without_the_word_pig():
    assert is_animal_study(
        "Joint degradation in a porcine model of meniscal injury",
        "Porcine stifle joints were examined.",
    )
