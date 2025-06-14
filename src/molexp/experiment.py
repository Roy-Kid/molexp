from pydantic import BaseModel, Field


class Experiment(BaseModel):
    """
    Represents an experiment with a name, description, and optional tags.
    """
    name: str
    readme: str = ""

    def __post_init__(self):
        """
        Post-initialization hook to set default values or perform additional setup.
        """