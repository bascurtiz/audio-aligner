-- elastique_worker.lua
-- Pitch-preserving warp using REAPER's élastique 3.3.3 Pro + stretch markers.
--
-- Env: ALIGN_CHECKER_REAPER_JOB = full path to JSON job file
-- On success writes <job>.done ; on failure writes <job>.error
--
-- Job JSON:
-- {
--   "jobs": [
--     {
--       "input": "C:/temp/padded.wav",
--       "output": "C:/temp/rendered.wav",
--       "target_length_sec": 294.8,
--       "sample_rate": 44100,
--       "markers": [ {"src": 0.0, "dst": 0.0}, {"src": 10.0, "dst": 10.05} ],
--       "stretch_mode": "transient"
--     }
--   ]
-- }

local function console(msg)
  if reaper and reaper.ShowConsoleMsg then
    reaper.ShowConsoleMsg(tostring(msg) .. "\n")
  end
end

local function write_text(path, text)
  local f = io.open(path, "wb")
  if not f then return false end
  f:write(text or "")
  f:close()
  return true
end

local function file_exists(path)
  local f = io.open(path, "rb")
  if not f then return false end
  f:close()
  return true
end

local function sleep(sec)
  local t0 = reaper.time_precise()
  while reaper.time_precise() - t0 < sec do
    -- spin
  end
end

local function wait_for_file(path, timeout)
  local t0 = reaper.time_precise()
  while reaper.time_precise() - t0 < timeout do
    if file_exists(path) then
      -- wait until size stable
      sleep(0.3)
      return true
    end
    -- REAPER may write without extension or with .wav forced
    local alt = {
      path,
      path:gsub("%.%w+$", ".wav"),
      path:gsub("%.%w+$", ".flac"),
      path .. ".wav",
    }
    for _, p in ipairs(alt) do
      if file_exists(p) then
        if p ~= path then
          -- copy/rename to expected path when possible
          local inf = io.open(p, "rb")
          local data = inf:read("*a")
          inf:close()
          local outf = io.open(path, "wb")
          if outf then
            outf:write(data)
            outf:close()
          end
        end
        sleep(0.2)
        return file_exists(path)
      end
    end
    sleep(0.2)
  end
  return false
end

----------------------------------------------------------------
-- Tiny JSON parser
----------------------------------------------------------------
local function parse_json(str)
  local pos = 1
  local function skip()
    while pos <= #str do
      local c = str:sub(pos, pos)
      if c ~= " " and c ~= "\t" and c ~= "\n" and c ~= "\r" then break end
      pos = pos + 1
    end
  end
  local parse_value
  local function parse_string()
    pos = pos + 1
    local out = {}
    while pos <= #str do
      local c = str:sub(pos, pos)
      if c == '"' then
        pos = pos + 1
        return table.concat(out)
      elseif c == "\\" then
        local n = str:sub(pos + 1, pos + 1)
        local map = { n = "\n", t = "\t", r = "\r", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
        out[#out + 1] = map[n] or n
        pos = pos + 2
      else
        out[#out + 1] = c
        pos = pos + 1
      end
    end
    error("unterminated string")
  end
  local function parse_number()
    local s, e = str:find("^%-?%d+%.?%d*[eE]?[%+%-]?%d*", pos)
    local n = tonumber(str:sub(s, e))
    pos = e + 1
    return n
  end
  local function parse_array()
    pos = pos + 1
    local arr = {}
    skip()
    if str:sub(pos, pos) == "]" then pos = pos + 1; return arr end
    while true do
      arr[#arr + 1] = parse_value()
      skip()
      local c = str:sub(pos, pos)
      if c == "]" then pos = pos + 1; break
      elseif c == "," then pos = pos + 1; skip()
      else error("bad array at " .. pos) end
    end
    return arr
  end
  local function parse_object()
    pos = pos + 1
    local obj = {}
    skip()
    if str:sub(pos, pos) == "}" then pos = pos + 1; return obj end
    while true do
      skip()
      local key = parse_string()
      skip()
      if str:sub(pos, pos) ~= ":" then error("expected :") end
      pos = pos + 1
      skip()
      obj[key] = parse_value()
      skip()
      local c = str:sub(pos, pos)
      if c == "}" then pos = pos + 1; break
      elseif c == "," then pos = pos + 1
      else error("bad object at " .. pos) end
    end
    return obj
  end
  parse_value = function()
    skip()
    local c = str:sub(pos, pos)
    if c == '"' then return parse_string()
    elseif c == "{" then return parse_object()
    elseif c == "[" then return parse_array()
    elseif str:sub(pos, pos + 3) == "true" then pos = pos + 4; return true
    elseif str:sub(pos, pos + 4) == "false" then pos = pos + 5; return false
    elseif str:sub(pos, pos + 3) == "null" then pos = pos + 4; return nil
    else return parse_number() end
  end
  return parse_value()
end

-- Synchronized follows the stretch markers.
-- Mode 9, submode 16: élastique 3.3.3 Pro, Synchronized: Normal.
-- Both stems use this engine. The instrumental also sets Transient-optimized.
local ELASTIQUE_33_PRO = (9 << 16) | 16

local function wipe_project()
  while reaper.CountMediaItems(0) > 0 do
    local it = reaper.GetMediaItem(0, 0)
    reaper.DeleteTrackMediaItem(reaper.GetMediaItem_Track(it), it)
  end
  while reaper.CountTracks(0) > 0 do
    reaper.DeleteTrack(reaper.GetTrack(0, 0))
  end
end

local function split_path(path)
  path = path:gsub("\\", "/")
  local dir, name = path:match("^(.*)/([^/]+)$")
  if not dir then
    return ".", path
  end
  local stem, ext = name:match("^(.*)(%.[^%.]+)$")
  if not stem then
    stem, ext = name, ""
  end
  return dir, stem, ext, name
end

local function process_job(job)
  local input = job.input
  local output = job.output
  local target_len = tonumber(job.target_length_sec)
  local markers = job.markers or {}
  local srate = tonumber(job.sample_rate) or 0

  if type(input) ~= "string" or input == "" then error("missing input") end
  if type(output) ~= "string" or output == "" then error("missing output") end
  if not target_len or target_len <= 0 then error("missing target_length_sec") end
  if not file_exists(input) then error("input missing: " .. input) end

  wipe_project()
  reaper.InsertTrackAtIndex(0, true)
  local track = reaper.GetTrack(0, 0)
  reaper.SetOnlyTrackSelected(track)

  -- Avoid muted/master issues
  reaper.SetMediaTrackInfo_Value(track, "D_VOL", 1.0)

  local before = reaper.CountMediaItems(0)
  reaper.InsertMedia(input, 0)
  if reaper.CountMediaItems(0) <= before then
    error("InsertMedia failed: " .. input)
  end

  local item = reaper.GetMediaItem(0, 0)
  local take = reaper.GetActiveTake(item)
  if not take then error("no active take") end

  -- De-click after loudness plays the file through RX only.
  -- Élastique here would stretch it a second time.
  local declick_only = job.declick_only == true
  if not declick_only then
    reaper.SetMediaItemTakeInfo_Value(take, "I_PITCHMODE", ELASTIQUE_33_PRO)
    reaper.SetMediaItemTakeInfo_Value(take, "B_PPITCH", 1)
  end

  local nmark = reaper.GetTakeNumStretchMarkers(take)
  for i = nmark - 1, 0, -1 do
    reaper.DeleteTakeStretchMarkers(take, i)
  end

  reaper.SetMediaItemInfo_Value(item, "D_POSITION", 0.0)
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH", target_len)
  -- Inserted items loop by default. If the stretch reads the source out
  -- before this length, the tail would replay the start of the file.
  -- Silence there instead. Same item path for acapella and instrumental.
  reaper.SetMediaItemInfo_Value(item, "B_LOOPSRC", 0)

  table.sort(markers, function(a, b)
    return (tonumber(a.dst) or 0) < (tonumber(b.dst) or 0)
  end)

  if not declick_only then
    for _, m in ipairs(markers) do
      local dst = math.max(0, math.min(target_len, tonumber(m.dst) or 0))
      local src = math.max(0, tonumber(m.src) or 0)
      reaper.SetTakeStretchMarker(take, -1, dst, src)
    end
  end

  -- Instrumental only. This is Project Settings > Stretch marker mode:
  -- Transient-optimized. 42337 is "Item: Force transient-optimized mode".
  -- The acapella stays on the project default.
  if job.stretch_mode == "transient" then
    reaper.SetMediaItemTakeInfo_Value(take, "I_STRETCHFLAGS", 4)
    reaper.SelectAllMediaItems(0, false)
    reaper.SetMediaItemSelected(item, true)
    reaper.Main_OnCommand(42337, 0)
  end

  reaper.UpdateItemInProject(item)
  reaper.UpdateArrange()

  -- Acapella only: RX 11 De-click "Fix Discontinuous Waveform"
  -- (multi-band random clicks, sensitivity 5, widening 4 ms).
  local declick = job.declick_chunk
  if type(declick) == "string" and declick ~= "" then
    local fx = reaper.TrackFX_AddByName(track, "VST3: RX 11 De-click (iZotope)", false, -1)
    if not fx or fx < 0 then
      error("RX 11 De-click (iZotope) is not installed")
    end
    reaper.TrackFX_Show(track, fx, 0)
    local loaded = reaper.TrackFX_SetNamedConfigParm(track, fx, "vst_chunk", declick)
    if not loaded then
      error("could not load RX De-click preset Fix Discontinuous Waveform")
    end
  end

  local out_dir, out_stem, out_ext = split_path(output)
  -- Ensure output directory exists (Python usually created it)
  -- RENDER_FILE = directory, RENDER_PATTERN = filename (no path)
  reaper.GetSetProjectInfo(0, "PROJECT_LENGTH", target_len, true)
  if srate > 0 then
    reaper.GetSetProjectInfo(0, "RENDER_SRATE", srate, true)
    reaper.SetCurrentBPM(0, reaper.Master_GetTempo(), false) -- no-op keep
  end
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 0, true) -- custom
  reaper.GetSetProjectInfo(0, "RENDER_STARTPOS", 0.0, true)
  reaper.GetSetProjectInfo(0, "RENDER_ENDPOS", target_len, true)
  reaper.GetSetProjectInfo(0, "RENDER_SETTINGS", 0, true) -- master mix
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 2, true)
  reaper.GetSetProjectInfo(0, "RENDER_ADDTOPROJ", 0, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_FILE", out_dir:gsub("/", "\\"), true)
  reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", out_stem, true)

  console("Rendering to dir=" .. out_dir .. " pattern=" .. out_stem)

  -- Delete stale outputs
  local stale = {
    output,
    out_dir .. "/" .. out_stem .. ".wav",
    out_dir .. "/" .. out_stem .. ".flac",
    out_dir .. "\\" .. out_stem .. ".wav",
    out_dir .. "\\" .. out_stem .. ".flac",
  }
  for _, p in ipairs(stale) do
    os.remove(p)
  end

  -- Render project using most recent render settings
  reaper.Main_OnCommand(42230, 0)

  local render_wait = 120
  if type(declick) == "string" and declick ~= "" then
    render_wait = 360
  end
  if not wait_for_file(output, render_wait) then
    -- try wav next to pattern
    local wav = out_dir .. "/" .. out_stem .. ".wav"
    if file_exists(wav) then
      local inf = io.open(wav, "rb")
      local data = inf:read("*a")
      inf:close()
      local outf = io.open(output, "wb")
      if not outf then error("cannot write expected output: " .. output) end
      outf:write(data)
      outf:close()
    else
      error("render produced no file at " .. output .. " (also missing " .. wav .. ")")
    end
  end

  console("OK " .. output)
end

----------------------------------------------------------------
local json_path = os.getenv("ALIGN_CHECKER_REAPER_JOB") or ""
if json_path == "" then
  local src = debug.getinfo(1, "S").source
  if src:sub(1, 1) == "@" then src = src:sub(2) end
  local dir = src:match("^(.*)[/\\]") or "."
  json_path = dir .. "/reaper_job.json"
end

local ok, err = pcall(function()
  console("align-checker elastique worker")
  console("job: " .. json_path)
  local f = io.open(json_path, "rb")
  if not f then error("cannot open job file") end
  local data = f:read("*a")
  f:close()
  local root = parse_json(data)
  local jobs = root.jobs or { root }
  for i, job in ipairs(jobs) do
    console(string.format("job %d/%d", i, #jobs))
    process_job(job)
  end
end)

if ok then
  write_text(json_path .. ".done", "ok\n")
else
  console("ERROR: " .. tostring(err))
  write_text(json_path .. ".error", tostring(err) .. "\n")
end

reaper.Main_OnCommand(40004, 0) -- close project
