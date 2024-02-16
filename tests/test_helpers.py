import pytest

import jaxrts


def test_dictionary_inversion():
    test_dict = {"a": 1, "b": 9, 2: "c"}
    assert jaxrts.helpers.invert_dict(test_dict) == {1: "a", 9: "b", "c": 2}


def test_dictionary_inversion_error_with_identical_keys():
    with pytest.raises(ValueError) as context:
        test_dict = {"a": 1, "b": 1, 2: "c"}
        jaxrts.helpers.invert_dict(test_dict)

    assert str(test_dict) in str(context.value)
    assert "cannot be inverted" in str(context.value)
