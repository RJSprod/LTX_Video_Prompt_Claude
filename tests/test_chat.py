"""Conversation mode, below the window.

Four things are checked here, and they are the four that would silently ruin a
chat: a character file that does not survive a round trip, an import that drops
a field, a history operation that loses a message, and a request built for the
model that has the wrong thing in it.
"""

from __future__ import annotations

import base64
import json

import pytest

from prompt_master.chat import prompt, yamlish
from prompt_master.chat.characters import (Character, CharacterStore, Persona, load_persona,
                                           read_card, safe_stem, save_persona)
from prompt_master.chat.history import ASSISTANT, USER, ChatStore, Conversation, Message
from prompt_master.core.paths import AppPaths


@pytest.fixture
def paths(tmp_path):
    made = AppPaths(tmp_path)
    made.create_managed_dirs()
    return made


@pytest.fixture
def store(paths):
    return CharacterStore.from_paths(paths)


@pytest.fixture
def chats(paths):
    return ChatStore.from_paths(paths)


# ── the file format ──────────────────────────────────────────────────────────

def test_a_character_file_survives_a_round_trip():
    written = yamlish.dumps({"name": "Ada", "context": "A runner.\n\nShe is fast.",
                             "greeting": "Hello: friend", "temperature": 0.9})
    read = yamlish.loads(written)
    assert read["name"] == "Ada"
    assert read["context"] == "A runner.\n\nShe is fast."
    # A colon inside a value is what quoting exists for.
    assert read["greeting"] == "Hello: friend"
    assert read["temperature"] == "0.9"


def test_the_block_scalars_oobabooga_writes_are_read():
    text = ("# a comment\n"
            "name: Chiharu Yamada\n"
            "context: |-\n"
            "  Chiharu is a computer engineer.\n"
            "\n"
            "  She likes motherboards.\n"
            "greeting: |\n"
            "  *strides in*\n"
            "  Hey!\n")
    read = yamlish.loads(text)
    assert read["name"] == "Chiharu Yamada"
    assert read["context"] == "Chiharu is a computer engineer.\n\nShe likes motherboards."
    assert read["greeting"] == "*strides in*\nHey!"


def test_a_folded_scalar_folds_and_a_quoted_one_unquotes():
    read = yamlish.loads("context: >-\n  one\n  two\n\n  three\nname: \"Ada: the second\"\n")
    assert read["context"] == "one two\nthree"
    assert read["name"] == "Ada: the second"


def test_a_value_that_would_read_back_as_something_else_is_quoted():
    for value in ("No", "true", "12", "1.5", "- dash", "#hash", ""):
        assert yamlish.loads(yamlish.dumps({"name": value}))["name"] == value


def test_a_nested_block_is_skipped_rather_than_guessed_at():
    read = yamlish.loads("name: Ada\nextensions:\n  depth: 3\n  world: none\ngreeting: hi\n")
    assert read["name"] == "Ada" and read["greeting"] == "hi"
    assert read["extensions"] == ""


# ── characters ───────────────────────────────────────────────────────────────

def test_a_saved_character_comes_back_the_same(store):
    original = Character(name="Ada", context="A runner.", greeting="Hi.",
                         temperature=0.7, top_p=0.8, max_reply_tokens=256, seed=11,
                         system="Be brief.")
    store.save(original)
    assert store.load("Ada") == original
    assert store.names() == ["Ada"]


def test_a_character_written_by_oobabooga_loads(store):
    (store.directory / "Chiharu.yaml").write_text(
        "name: Chiharu Yamada\ncontext: |-\n  A computer engineer.\ngreeting: Hey!\n",
        encoding="utf-8")
    loaded = store.load("Chiharu Yamada")
    assert (loaded.name, loaded.context, loaded.greeting) == (
        "Chiharu Yamada", "A computer engineer.", "Hey!")
    # The settings this application adds open on their defaults rather than
    # refusing a file that has never heard of them.
    assert loaded.temperature == Character(name="x").temperature


def test_renaming_a_character_moves_its_file_and_its_picture(store, tmp_path):
    from PIL import Image

    store.save(Character(name="Ada", context="A runner."))
    picture = tmp_path / "face.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    store.set_avatar("Ada", picture)
    assert store.avatar_for("Ada") is not None

    store.save(Character(name="Ada Lovelace", context="A runner."), previous_name="Ada")
    assert store.names() == ["Ada Lovelace"]
    assert store.avatar_for("Ada Lovelace") is not None
    assert store.avatar_for("Ada") is None
    assert not store.path_for("Ada").exists()


def test_deleting_a_character_takes_its_picture_with_it(store, tmp_path):
    from PIL import Image

    store.save(Character(name="Ada"))
    picture = tmp_path / "face.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    store.set_avatar("Ada", picture)
    store.delete("Ada")
    assert store.names() == [] and store.avatar_for("Ada") is None


def test_a_broken_file_does_not_empty_the_list(store):
    store.save(Character(name="Ada"))
    (store.directory / "broken.yaml").write_bytes(b"\xff\xfe not text at all")
    assert store.names() == ["Ada"]


def test_a_name_that_is_not_a_file_name_still_saves(store):
    store.save(Character(name='Ada / "The Runner" *'))
    assert store.names() == ['Ada / "The Runner" *']
    assert ":" not in safe_stem('Ada: the runner')


# ── importing ────────────────────────────────────────────────────────────────

def test_importing_a_tavern_json_folds_every_field_into_the_context(store, tmp_path):
    source = tmp_path / "card.json"
    source.write_text(json.dumps({
        "char_name": "Ada", "char_persona": "A runner.", "char_greeting": "Hi.",
        "personality": "Impatient.", "world_scenario": "A bridge at night.",
        "example_dialogue": "You: hello\nAda: hello",
    }), encoding="utf-8")

    imported = store.import_file(source)
    assert imported.name == "Ada" and imported.greeting == "Hi."
    for expected in ("A runner.", "Impatient.", "A bridge at night.", "Ada: hello"):
        assert expected in imported.context
    assert store.load("Ada").context == imported.context


def test_importing_a_png_character_card_reads_it_and_keeps_the_picture(store, tmp_path):
    from PIL import Image, PngImagePlugin

    card = {"spec": "chara_card_v2",
            "data": {"name": "Chiharu", "description": "An engineer.", "first_mes": "Hey!"}}
    info = PngImagePlugin.PngInfo()
    info.add_text("chara", base64.b64encode(json.dumps(card).encode()).decode())
    source = tmp_path / "chiharu.png"
    Image.new("RGB", (16, 16), "blue").save(source, pnginfo=info)

    imported = store.import_file(source)
    assert (imported.name, imported.context, imported.greeting) == (
        "Chiharu", "An engineer.", "Hey!")
    assert store.avatar_for("Chiharu") is not None


def test_a_plain_image_is_refused_rather_than_imported_empty(store, tmp_path):
    from PIL import Image

    source = tmp_path / "holiday.png"
    Image.new("RGB", (8, 8), "green").save(source)
    with pytest.raises(ValueError):
        read_card(source)
    with pytest.raises(ValueError):
        store.import_file(source)


def test_importing_the_same_character_twice_does_not_overwrite_the_first(store, tmp_path):
    source = tmp_path / "ada.yaml"
    source.write_text("name: Ada\ncontext: A runner.\n", encoding="utf-8")
    store.import_file(source)
    second = store.import_file(source)
    assert second.name == "Ada (2)"
    assert store.names() == ["Ada", "Ada (2)"]


# ── the persona ──────────────────────────────────────────────────────────────

def test_an_undefined_persona_is_an_unnamed_you(paths):
    persona = load_persona(paths)
    assert not persona.defined and persona.display == "You"


def test_a_defined_persona_is_remembered(paths):
    save_persona(paths, Persona(name="Rashan", description="Writes prompts."))
    reloaded = load_persona(paths)
    assert reloaded.defined and reloaded.display == "Rashan"
    assert reloaded.description == "Writes prompts."


# ── history ──────────────────────────────────────────────────────────────────

def test_regenerating_keeps_the_reply_it_replaced(chats):
    conversation = chats.new("Ada")
    conversation.append(USER, "hello")
    reply = conversation.append(ASSISTANT, "first try")
    reply.add_version("second try")

    assert reply.text == "second try" and len(reply.versions) == 2
    reply.show(0)
    assert reply.text == "first try"
    reply.drop_version()
    assert reply.versions == ["second try"] and reply.text == "second try"
    reply.drop_version()                       # the last one is never dropped
    assert reply.versions == ["second try"]


def test_branching_copies_up_to_the_message_and_leaves_the_original(chats):
    conversation = chats.new("Ada")
    for index in range(4):
        conversation.append(USER if index % 2 == 0 else ASSISTANT, f"line {index}")
    chats.save(conversation)

    branched = chats.branch(conversation, 1)
    assert [message.text for message in branched.messages] == ["line 0", "line 1"]
    assert len(conversation.messages) == 4          # untouched
    assert branched.identifier != conversation.identifier
    assert {row.identifier for row in chats.listing("Ada")} == {
        conversation.identifier, branched.identifier}
    # The copy is a copy: editing one cannot reach the other.
    branched.messages[0].text = "changed"
    assert conversation.messages[0].text == "line 0"


def test_deleting_from_a_message_takes_everything_after_it(chats):
    conversation = chats.new("Ada")
    for index in range(5):
        conversation.append(USER, f"line {index}")
    conversation.delete_from(2)
    assert [message.text for message in conversation.messages] == ["line 0", "line 1"]
    conversation.truncate_after(0)
    assert [message.text for message in conversation.messages] == ["line 0"]
    conversation.delete(0)
    assert conversation.messages == []


def test_a_chat_is_named_after_the_first_thing_said_in_it(chats):
    conversation = chats.new("Ada")
    conversation.append(ASSISTANT, "A greeting the character always opens with")
    assert conversation.title == "New chat"     # the greeting is not your line
    conversation.append(USER, "  tell me about   the bridge  ")
    assert conversation.title == "tell me about the bridge"


def test_a_saved_chat_reloads_with_its_versions_and_pictures(chats):
    conversation = chats.new("Ada")
    conversation.append(USER, "look at this", image="data:image/jpeg;base64,AAAA",
                        image_name="still.png")
    reply = conversation.append(ASSISTANT, "one")
    reply.add_version("two")
    reply.show(0)
    chats.save(conversation)

    reloaded = chats.load("Ada", conversation.identifier)
    assert reloaded.messages[0].image_name == "still.png"
    assert reloaded.messages[0].image.startswith("data:image/jpeg;base64,")
    assert reloaded.messages[1].versions == ["one", "two"] and reloaded.messages[1].text == "one"


def test_chats_are_listed_newest_first_and_deleted_by_name(chats):
    first = chats.new("Ada")
    first.title = "older"
    chats.save(first)
    second = chats.new("Ada")
    second.title = "newer"
    chats.save(second)
    second.updated = first.updated + 100
    chats.save(second)

    assert [row.title for row in chats.listing("Ada")][0] == "newer"
    chats.delete("Ada", second.identifier)
    assert [row.title for row in chats.listing("Ada")] == ["older"]


def test_the_start_of_a_reply_is_kept_with_the_chat(chats):
    """It is written into one conversation, so it belongs to that conversation
    and to no other — including the next one with the same character."""
    conversation = chats.new("Ada")
    conversation.response_prefix = "It's always been red"
    chats.save(conversation)

    assert chats.load("Ada", conversation.identifier).response_prefix == "It's always been red"
    assert chats.new("Ada").response_prefix == ""        # a new chat starts with none


def test_a_chat_written_before_there_were_starts_simply_has_none(chats):
    written = chats.new("Ada").to_dict()
    del written["response_prefix"]
    assert Conversation.from_dict(written).response_prefix == ""
    assert Conversation.from_dict({"id": "x", "character": "Ada",
                                   "response_prefix": None}).response_prefix == ""


def test_a_branch_carries_the_start_into_the_copy(chats):
    """A branch carries on from a point in the chat, and the start the replies
    were being given is part of what that point is."""
    conversation = chats.new("Ada")
    conversation.append(USER, "one")
    conversation.append(ASSISTANT, "two")
    conversation.response_prefix = "It's always been red"
    chats.save(conversation)

    branched = chats.branch(conversation, 1)
    assert branched.response_prefix == "It's always been red"
    # And the two are separate from then on.
    branched.response_prefix = "It's always been blue"
    chats.save(branched)
    assert chats.load("Ada", conversation.identifier).response_prefix == "It's always been red"


def test_two_characters_do_not_share_a_chat_list(chats):
    chats.save(chats.new("Ada"))
    assert chats.listing("Ada") and chats.listing("Chiharu") == []


# ── what goes on the wire ────────────────────────────────────────────────────

def _messages(*pairs):
    return [Message(role=role, versions=[text]) for role, text in pairs]


def test_the_system_message_carries_the_character_and_the_placeholders():
    character = Character(name="Ada", context="{{char}} always calls {{user}} by name.")
    system = prompt.system_text(character, Persona(name="Rashan", description="A director."))
    assert "You are Ada" in system
    assert "Ada always calls Rashan by name." in system
    assert "You are talking to Rashan" in system and "A director." in system
    assert "{{char}}" not in system and "{{user}}" not in system


def test_an_undefined_persona_says_nothing_about_you():
    system = prompt.system_text(Character(name="Ada", context="A runner."), Persona())
    assert "You are talking to" not in system
    assert "You are Ada" in system and "A runner." in system


def test_a_custom_system_message_replaces_the_wrapper():
    character = Character(name="Ada", context="A runner.", system="Answer as {{char}}, tersely.")
    system = prompt.system_text(character, Persona())
    assert system == "Answer as Ada, tersely."
    assert "A runner." not in system


def test_the_greeting_is_substituted_too():
    character = Character(name="Ada", greeting="Hello {{user}}, I am {{char}}.")
    assert prompt.greeting_text(character, Persona(name="Rashan")) == "Hello Rashan, I am Ada."


def test_the_request_is_the_system_message_then_the_turns():
    built = prompt.build(Character(name="Ada", context="A runner."), Persona(),
                         _messages((USER, "hello"), (ASSISTANT, "hi"), (USER, "how are you")))
    assert [message["role"] for message in built] == ["system", USER, ASSISTANT, USER]
    assert built[0]["content"].startswith("You are Ada")
    assert built[-1]["content"] == "how are you"


def test_an_instruction_is_the_last_turn_and_is_not_stored():
    history = _messages((USER, "hello"), (ASSISTANT, "half a repl"))
    built = prompt.build(Character(name="Ada"), Persona(), history,
                         instruction=prompt.continue_instruction(Character(name="Ada")))
    assert built[-1]["role"] == USER and "Continue Ada's last message" in built[-1]["content"]
    assert len(history) == 2                     # the instruction went nowhere near it


def test_a_started_reply_goes_out_as_the_assistant_turn_it_already_is():
    """The start is on the wire as the opening of the reply, and the instruction
    that follows asks for the rest of it. That shape is what a continuation
    uses, because llama-server closes an assistant turn it is given rather than
    writing on from it."""
    history = _messages((USER, "What colour is the sky"),
                        (ASSISTANT, "It's always been red"))
    built = prompt.build(Character(name="Ada"), Persona(), history,
                         instruction=prompt.prefix_instruction(Character(name="Ada")))

    assert [message["role"] for message in built] == ["system", USER, ASSISTANT, USER]
    assert built[-2]["content"] == "It's always been red"
    instruction = built[-1]["content"]
    assert "already been started" in instruction
    assert "Carry straight on" in instruction and "Ada's voice" in instruction
    # A start is written precisely because the character would not have chosen
    # it, so being told to continue is not enough on its own.
    for held in ("accept whatever", "even where you would have said something else",
                 "do not contradict or walk it back", "Do not repeat it"):
        assert held.casefold() in instruction.casefold()


def test_a_message_with_a_picture_takes_the_multimodal_shape():
    history = [Message(role=USER, versions=["what is this"],
                       image="data:image/jpeg;base64,AAAA", image_name="still.png")]
    built = prompt.build(Character(name="Ada"), Persona(), history)
    content = built[-1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "what is this"}
    assert prompt.has_image(history)


def test_only_the_most_recent_pictures_are_carried():
    history = [Message(role=USER, versions=[f"line {index}"], image="data:image/jpeg;base64,AAAA",
                       image_name=f"{index}.png") for index in range(prompt.MAX_IMAGES + 2)]
    built = prompt.build(Character(name="Ada"), Persona(), history, context_size=32000)
    parts = [message["content"] for message in built[1:]]
    assert sum(1 for part in parts if isinstance(part, list)) == prompt.MAX_IMAGES
    # The ones that lost their picture still say there was one.
    assert "[image: 0.png]" in parts[0]


def test_a_long_conversation_is_trimmed_from_the_front():
    history = _messages(*[(USER if index % 2 == 0 else ASSISTANT, f"line {index} " + "x" * 400)
                          for index in range(60)])
    built = prompt.build(Character(name="Ada", context="A runner."), Persona(),
                         history, context_size=2048, reply_tokens=256)
    assert built[0]["role"] == "system"                    # never dropped
    assert len(built) < len(history)                       # something was
    assert "line 59" in built[-1]["content"]               # the newest survives


def test_the_last_message_is_sent_even_when_it_cannot_possibly_fit():
    history = _messages((USER, "y" * 20000))
    built = prompt.build(Character(name="Ada"), Persona(), history, context_size=512)
    assert len(built) == 2 and built[-1]["content"].startswith("yyy")


def test_a_name_label_the_model_wrote_anyway_is_stripped():
    character, persona = Character(name="Ada"), Persona(name="Rashan")
    assert prompt.clean_reply("Ada: hello there", character, persona) == "hello there"
    assert prompt.clean_reply("  hello there", character, persona) == "hello there"
    # A colon that is not a label is left alone.
    assert prompt.clean_reply("here it is: a bridge", character, persona) == "here it is: a bridge"


def test_a_character_whose_file_is_not_named_after_it_is_edited_in_place(store):
    """oobabooga's own example is ``Chiharu.yaml`` holding "Chiharu Yamada". A
    save has to go back into that file rather than beside it."""
    (store.directory / "Chiharu.yaml").write_text(
        "name: Chiharu Yamada\ncontext: An engineer.\n", encoding="utf-8")
    loaded = store.load("Chiharu Yamada")
    loaded.temperature = 0.4
    store.save(loaded)

    assert store.names() == ["Chiharu Yamada"]
    assert not (store.directory / "Chiharu Yamada.yaml").exists()
    assert store.load("Chiharu Yamada").temperature == 0.4
    assert store.unique_name("Chiharu Yamada") == "Chiharu Yamada (2)"
