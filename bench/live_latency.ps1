$url = "https://performances-than-turning-dance.trycloudflare.com"
$key = python -c "from dotenv import dotenv_values; print(dotenv_values('.env')['API_KEY'])"
$h = @{ "X-API-Key" = $key }
$up = curl.exe -s -H "X-API-Key: $key" -F "file=@data\images\test1.jpg" "$url/detect" | ConvertFrom-Json
for ($i = 1; $i -le 30; $i++) {
    $body = @{ image_id = $up.image_id; question = "How many cars are in this image?" } | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Uri "$url/ask" -Method Post -Headers $h -ContentType "application/json" -Body $body
        Write-Host "$i status=$($r.status)"
    } catch {
        Write-Host "$i error=$($_.Exception.Response.StatusCode.value__)"
    }
    Start-Sleep -Seconds 4
}