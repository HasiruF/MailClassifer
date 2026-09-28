from pydantic import BaseModel


class GmailTokenResponse(BaseModel):
    access_token: str
    expires_in: int
