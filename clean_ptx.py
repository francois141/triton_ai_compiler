import re


DEBUG_DIRECTIVE_PATTERN = re.compile(r"^\s*\.(?:file|loc)\b")
SECTION_DIRECTIVE_PATTERN = re.compile(r"^\s*\.section\b")
FUNCTION_DIRECTIVE_PATTERN = re.compile(
    r"^\s*(?:\.(?:visible|weak|extern)\s+)*\.(?:entry|func)\b"
)


def clean_ptx(ptx):
    cleaned = []
    index = 0
    in_block_comment = False
    in_string = False
    escaped = False

    while index < len(ptx):
        character = ptx[index]
        next_character = ptx[index + 1] if index + 1 < len(ptx) else ""

        if in_block_comment:
            if character == "*" and next_character == "/":
                in_block_comment = False
                index += 2
                continue
            if character == "\n":
                cleaned.append(character)
            index += 1
            continue

        if in_string:
            cleaned.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            cleaned.append(character)
            index += 1
        elif character == "/" and next_character == "/":
            newline_index = ptx.find("\n", index)
            if newline_index == -1:
                break
            cleaned.append("\n")
            index = newline_index + 1
        elif character == "/" and next_character == "*":
            in_block_comment = True
            index += 2
        else:
            cleaned.append(character)
            index += 1

    output = []
    section_depth = 0
    expects_function_body = False

    for line in "".join(cleaned).splitlines(keepends=True):
        if section_depth:
            section_depth += line.count("{") - line.count("}")
            continue

        if SECTION_DIRECTIVE_PATTERN.match(line):
            section_depth = line.count("{") - line.count("}")
            continue

        if FUNCTION_DIRECTIVE_PATTERN.match(line):
            expects_function_body = True

        if line.lstrip().startswith("{"):
            if expects_function_body:
                expects_function_body = False
            else:
                section_depth = line.count("{") - line.count("}")
                continue

        if line.strip() and not DEBUG_DIRECTIVE_PATTERN.match(line):
            output.append(line)

    return "".join(output)
