param(
  [int]$Port = 5173,
  [string]$Root = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$rootPath = [System.IO.Path]::GetFullPath($Root)
if (-not $rootPath.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
  $rootPath += [System.IO.Path]::DirectorySeparatorChar
}

$mimeTypes = @{
  ".html" = "text/html; charset=utf-8"
  ".htm" = "text/html; charset=utf-8"
  ".css" = "text/css; charset=utf-8"
  ".js" = "text/javascript; charset=utf-8"
  ".json" = "application/json; charset=utf-8"
  ".svg" = "image/svg+xml"
  ".png" = "image/png"
  ".jpg" = "image/jpeg"
  ".jpeg" = "image/jpeg"
  ".gif" = "image/gif"
  ".ico" = "image/x-icon"
  ".txt" = "text/plain; charset=utf-8"
}

function Send-Response {
  param(
    [System.Net.Sockets.NetworkStream]$Stream,
    [int]$Status,
    [string]$Reason,
    [byte[]]$Body,
    [string]$ContentType = "text/plain; charset=utf-8",
    [bool]$HeadOnly = $false
  )

  $headers = "HTTP/1.1 $Status $Reason`r`n" +
    "Content-Type: $ContentType`r`n" +
    "Content-Length: $($Body.Length)`r`n" +
    "Connection: close`r`n" +
    "Cache-Control: no-cache`r`n" +
    "`r`n"
  $headerBytes = [System.Text.Encoding]::ASCII.GetBytes($headers)
  $Stream.Write($headerBytes, 0, $headerBytes.Length)
  if (-not $HeadOnly -and $Body.Length -gt 0) {
    $Stream.Write($Body, 0, $Body.Length)
  }
}

function Get-Request {
  param([System.Net.Sockets.NetworkStream]$Stream)

  $buffer = New-Object byte[] 8192
  $builder = New-Object System.Text.StringBuilder
  do {
    $read = $Stream.Read($buffer, 0, $buffer.Length)
    if ($read -le 0) { break }
    [void]$builder.Append([System.Text.Encoding]::ASCII.GetString($buffer, 0, $read))
  } while (-not $builder.ToString().Contains("`r`n`r`n"))

  return $builder.ToString()
}

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $Port)
$listener.Start()

Write-Host "3D Swarm Pathfinding Lab"
Write-Host "Serving: $rootPath"
Write-Host "URL: http://127.0.0.1:$Port/"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

try {
  while ($true) {
    $client = $listener.AcceptTcpClient()
    try {
      $stream = $client.GetStream()
      $request = Get-Request -Stream $stream
      $firstLine = ($request -split "`r`n", 2)[0]
      $parts = $firstLine -split " "

      if ($parts.Count -lt 2 -or ($parts[0] -ne "GET" -and $parts[0] -ne "HEAD")) {
        $body = [System.Text.Encoding]::UTF8.GetBytes("Method not allowed")
        Send-Response -Stream $stream -Status 405 -Reason "Method Not Allowed" -Body $body
        continue
      }

      $path = [System.Uri]::UnescapeDataString(($parts[1] -split "\?", 2)[0])
      if ($path -eq "/") { $path = "/index.html" }
      $relativePath = $path.TrimStart("/") -replace "/", [System.IO.Path]::DirectorySeparatorChar
      $filePath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($rootPath, $relativePath))

      if (-not $filePath.StartsWith($rootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        $body = [System.Text.Encoding]::UTF8.GetBytes("Forbidden")
        Send-Response -Stream $stream -Status 403 -Reason "Forbidden" -Body $body -HeadOnly ($parts[0] -eq "HEAD")
        continue
      }

      if (-not [System.IO.File]::Exists($filePath)) {
        $body = [System.Text.Encoding]::UTF8.GetBytes("Not found")
        Send-Response -Stream $stream -Status 404 -Reason "Not Found" -Body $body -HeadOnly ($parts[0] -eq "HEAD")
        continue
      }

      $extension = [System.IO.Path]::GetExtension($filePath).ToLowerInvariant()
      $contentType = $mimeTypes[$extension]
      if (-not $contentType) { $contentType = "application/octet-stream" }

      $body = [System.IO.File]::ReadAllBytes($filePath)
      Send-Response -Stream $stream -Status 200 -Reason "OK" -Body $body -ContentType $contentType -HeadOnly ($parts[0] -eq "HEAD")
    } catch {
      try {
        $body = [System.Text.Encoding]::UTF8.GetBytes("Server error")
        Send-Response -Stream $stream -Status 500 -Reason "Internal Server Error" -Body $body
      } catch {}
    } finally {
      $client.Close()
    }
  }
} finally {
  $listener.Stop()
}
