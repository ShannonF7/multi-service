from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from src.core.config import settings
from src.scripts.schemas import BaseResponse, ErrorCode

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.API_KEY = settings.api_key
        # Allow multiple IPs separated by comma
        ips = settings.allowed_ips
        self.ALLOWED_IPS = [ip.strip() for ip in ips.split(",")] if ips else []

    async def dispatch(self, request: Request, call_next):
        # Skip security check for docs, openapi, and health check
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health"]:
            return await call_next(request)
        
        # Skip OPTIONS for CORS
        if request.method == "OPTIONS":
            return await call_next(request)

        # IP Check
        if self.ALLOWED_IPS:
            client_ip = request.client.host
            if client_ip not in self.ALLOWED_IPS:
                return JSONResponse(
                    status_code=403,
                    content=BaseResponse(
                        code=ErrorCode.FORBIDDEN.value,
                        message=f"IP {client_ip} is not allowed",
                        data=None
                    ).dict()
                )

        # API Key Check
        if self.API_KEY:
            api_key = request.headers.get("X-API-KEY")
            if api_key != self.API_KEY:
                return JSONResponse(
                    status_code=401,
                    content=BaseResponse(
                        code=ErrorCode.UNAUTHORIZED.value,
                        message="Invalid or missing API Key",
                        data=None
                    ).dict()
                )
        
        return await call_next(request)
