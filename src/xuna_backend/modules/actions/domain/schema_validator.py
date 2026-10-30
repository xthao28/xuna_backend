"""No-dependency JSON Schema subset validator for XTP tool descriptors (UDC-4 §5.2).

Supported keywords:
- Top-level: `type == "object"`, `properties` (`dict`), `required` (`list[str]`),
  `additionalProperties is False`.
- Property schemas:
  - `type == "string"`: optional `maxLength`, `pattern`, `enum`, `description`.
  - `type == "integer"`: optional `minimum`, `maximum`, `enum`, `description`.

Any other keyword or type in a descriptor schema is rejected with `SchemaDefinitionError`.
"""

import re

_ALLOWED_TOP_KEYS = frozenset({"type", "properties", "required", "additionalProperties"})
_ALLOWED_STRING_KEYS = frozenset({"type", "maxLength", "pattern", "enum", "description"})
_ALLOWED_INT_KEYS = frozenset({"type", "minimum", "maximum", "enum", "description"})


class SchemaDefinitionError(ValueError):
    """Raised when a tool descriptor's JSON Schema uses unsupported or unsafe keywords."""


class ArgumentValidationError(ValueError):
    """Raised when untrusted tool call arguments fail schema validation."""


class SchemaValidator:
    """Deterministic validator for the XTP JSON Schema subset."""

    @staticmethod
    def validate_schema_definition(schema: dict[str, object]) -> None:
        if not isinstance(schema, dict):
            raise SchemaDefinitionError("schema must be a dict")
        unknown_top = set(schema.keys()) - _ALLOWED_TOP_KEYS
        if unknown_top:
            raise SchemaDefinitionError(f"unsupported top-level schema keys: {sorted(unknown_top)}")
        if schema.get("type") != "object":
            raise SchemaDefinitionError("top-level schema type must be 'object'")
        if schema.get("additionalProperties") is not False:
            raise SchemaDefinitionError("top-level additionalProperties must be False")

        props = schema.get("properties", {})
        if not isinstance(props, dict):
            raise SchemaDefinitionError("properties must be a dict")

        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
            raise SchemaDefinitionError("required must be a list of strings")
        for req_key in required:
            if req_key not in props:
                raise SchemaDefinitionError(f"required key {req_key!r} not in properties")

        for prop_name, prop_schema in props.items():
            if not isinstance(prop_name, str) or not prop_name:
                raise SchemaDefinitionError("property name must be a non-empty string")
            if not isinstance(prop_schema, dict):
                raise SchemaDefinitionError(f"property {prop_name!r} schema must be a dict")
            SchemaValidator._validate_property_definition(prop_name, prop_schema)

    @staticmethod
    def _validate_property_definition(name: str, prop: dict[str, object]) -> None:
        prop_type = prop.get("type")
        if prop_type == "string":
            unknown = set(prop.keys()) - _ALLOWED_STRING_KEYS
            if unknown:
                raise SchemaDefinitionError(
                    f"unsupported string keywords on {name!r}: {sorted(unknown)}"
                )
            if "maxLength" in prop:
                val = prop["maxLength"]
                if type(val) is not int or val < 0:
                    raise SchemaDefinitionError(f"maxLength on {name!r} must be a non-negative int")
            if "pattern" in prop:
                pat = prop["pattern"]
                if not isinstance(pat, str):
                    raise SchemaDefinitionError(f"pattern on {name!r} must be a string")
                try:
                    re.compile(pat)
                except re.error as error:
                    raise SchemaDefinitionError(f"invalid regex on {name!r}") from error
            if "enum" in prop:
                enum_vals = prop["enum"]
                if (
                    not isinstance(enum_vals, list)
                    or not enum_vals
                    or not all(isinstance(v, str) for v in enum_vals)
                ):
                    raise SchemaDefinitionError(f"enum on {name!r} must be a non-empty str list")
        elif prop_type == "integer":
            unknown = set(prop.keys()) - _ALLOWED_INT_KEYS
            if unknown:
                raise SchemaDefinitionError(
                    f"unsupported integer keywords on {name!r}: {sorted(unknown)}"
                )
            for bound_key in ("minimum", "maximum"):
                if bound_key in prop and type(prop[bound_key]) is not int:
                    raise SchemaDefinitionError(f"{bound_key} on {name!r} must be an int")
            if "enum" in prop:
                enum_vals = prop["enum"]
                if (
                    not isinstance(enum_vals, list)
                    or not enum_vals
                    or not all(type(v) is int for v in enum_vals)
                ):
                    raise SchemaDefinitionError(f"enum on {name!r} must be a non-empty int list")
        else:
            raise SchemaDefinitionError(f"unsupported property type {prop_type!r} on {name!r}")

    @staticmethod
    def validate_arguments(
        schema: dict[str, object],
        arguments: dict[str, object],
        *,
        allow_target: bool = False,
    ) -> tuple[dict[str, object], str | None]:
        """Validates `arguments` against `schema` and extracts optional `target` when
        `allow_target=True`. Returns `(clean_args_without_target, requested_target)`."""
        SchemaValidator.validate_schema_definition(schema)
        if not isinstance(arguments, dict):
            raise ArgumentValidationError("arguments must be a JSON object")

        props_raw = schema.get("properties", {})
        props = props_raw if isinstance(props_raw, dict) else {}
        required_raw = schema.get("required", [])
        required = required_raw if isinstance(required_raw, list) else []

        clean_args: dict[str, object] = {}
        requested_target: str | None = None

        for arg_key, arg_val in arguments.items():
            if allow_target and arg_key == "target" and "target" not in props:
                if not isinstance(arg_val, str) or not 1 <= len(arg_val.strip()) <= 24:
                    raise ArgumentValidationError("target must be a non-empty host alias")
                requested_target = arg_val.strip()
                continue
            if arg_key not in props:
                raise ArgumentValidationError(f"unexpected argument: {arg_key}")
            prop_schema = props[arg_key]
            if not isinstance(prop_schema, dict):
                raise SchemaDefinitionError(f"property {arg_key!r} schema must be a dict")
            SchemaValidator._validate_value(arg_key, arg_val, prop_schema)
            clean_args[arg_key] = arg_val

        for req_key in required:
            if req_key not in clean_args:
                raise ArgumentValidationError(f"missing required argument: {req_key}")

        return clean_args, requested_target

    @staticmethod
    def _validate_value(name: str, value: object, prop: dict[str, object]) -> None:
        prop_type = prop.get("type")
        if prop_type == "string":
            if not isinstance(value, str):
                raise ArgumentValidationError(f"argument {name!r} must be a string")
            max_len = prop.get("maxLength")
            if type(max_len) is int and len(value) > max_len:
                raise ArgumentValidationError(f"argument {name!r} exceeds maxLength {max_len}")
            pattern = prop.get("pattern")
            if isinstance(pattern, str) and not re.search(pattern, value):
                raise ArgumentValidationError(f"argument {name!r} does not match pattern")
            enum_vals = prop.get("enum")
            if isinstance(enum_vals, list) and value not in enum_vals:
                raise ArgumentValidationError(f"argument {name!r} not in allowed enum")
        elif prop_type == "integer":
            # Note: bool is a subclass of int in Python; reject booleans explicitly.
            if type(value) is not int:
                raise ArgumentValidationError(f"argument {name!r} must be an integer")
            minimum = prop.get("minimum")
            if type(minimum) is int and value < minimum:
                raise ArgumentValidationError(f"argument {name!r} < minimum {minimum}")
            maximum = prop.get("maximum")
            if type(maximum) is int and value > maximum:
                raise ArgumentValidationError(f"argument {name!r} > maximum {maximum}")
            enum_vals = prop.get("enum")
            if isinstance(enum_vals, list) and value not in enum_vals:
                raise ArgumentValidationError(f"argument {name!r} not in allowed enum")
