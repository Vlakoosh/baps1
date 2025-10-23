SOUND BUTTON: 
when you press the sound button, you go into SOUND mode
pressing a matrix button/pad in SOUND mode lets you select one of your 16 kits/sounds.
It will show on the screen the sound file opened in that slot. 
You can use one of the knobs to change the sound file in the list. 
Slots 1-8 read from data/sounds while 9-16 read from data/kits

FREE MODE:
based on what sound/kit you select in the SOUND mode, that will be available in the FREE play mode.
Keep in mind that kits are only available from 9 to 16. 1 through 8 are melodic sound oneshots for stuff like chords, bass notes, etc. 
if you select a sound in 1-8, you get a 'keyboard' that changes the pitch of the sound. 
If you select 9-16, you get the kit with 16 sounds. This is like a drum pad.

PATTERN BUTTON: 
- holding the pattern button and clicking on one of 16 buttons will select that pattern. 

If you press play it will play that pattern on loop. 
if you hold the pattern button and then click more than one pattern, it will create a chain of patterns. 
You can do this in any order. For example, you can press patterns 1,2,3,4,5,6,7,8 one after another and it will play them in that order and loop. 
You can also press one pattern more than once and it will chain it as well. 
They also don't need to be in numerical order. You can for example do: 9,9,9, 10, 2, 6, 5, 9, 1 and it will play in that order and start from scratch on finish 

BPM BUTTON: pressing the bpm button will put you in BPM mode
- turning the wheel 2 in this mode will change the bpm/tempo. It will go from 60 to 300 bpm 
- turning the wheel 1 in BPM mode will change the swing amount 

PLAY BUTTON: 
plays the selected patterns xd (in a loop) 
pressing it again will pause all playback
this should work in basically any mode 

WRITE BUTTON: 
when you press the write button, you enter the write mode. 
The last sound you played in the current kit will be the sound that you write. 
This applies both to kits and oneshots. However, with oneshots you will instead sequence the last note you played. 

You can press on the buttons in write mode to toggle each note. 

RECORD BUTTON: 
when you press the record button, you go into RECORD mode
you can then hold one of the first 8 buttons to record a 'melodic' oneshot sample. 
That sample will be a one shot that you can use to play on the matrix like a piano. 

You can also hold one of the other 8 buttons (9-16) to record a 'kit' sample. 
That sample can have up to 16 sounds in it which will be cut up with start and end times for each additional sounds. This will be one continuous sample/sound

for sample trimming, you set a start time and end time. 
The end time is not relative to the start of the entire sample, but it's duration from the start time. 
for example, if you have a sample that you trim to start at 0:1:250 and end at 0:0:500, it doesn't go back in time, it just lasts for that time. 

VOLUME BUTTON:
when you press the volume button, you go into VOLUME mode
if you turn knob 1 in this mode, it will change the level of the whole kit.
if you turn knob 2 in this mode, it will change the level of the last sound you pressed. (with live sound preview on each tick of turning)
you can still press notes/pads in this mode to preview and 'select' the sounds.

TRIM BUTTON: 
when you press the TRIM button, you go into TRIM mode
if you are in kits 1-8 (one shots) it will trim that oneshot/sound.

If you are in kits (9-16), it will trim the last sound you pressed.
You can press another one to open it and trim it. 

It should show on the screen bar the region of the sound sample that you selected. 
Also, when you are moving the trim knobs, it will play the sound after every tick of the knob so as you turn it you can see how the sound changes. 

FX BUTTON:
out of scope for now.


pressing any button again will automatically go into free mode


TECHNICAL IMPLEMENTATION: 
- raspberry pi zero 
- SCREEN: i2c luma ssd1309. 
  - For local testing on pc we will use luma.emulator which uses the same commands but displays a pygame window instead.
- 5x5 button matrix. 4x4 pads section with the 9 remaining buttons being the MODE buttons.
- 2 knobs. For simplicity call them knob 1 and knob 2.


SCREEN details:
top left will always show the active mode and next to it the number of the last pressed pad.
top right will show the relevant number to the mode. In bpm mode show swing/bpm. in volume mode show pad/kit volumes etc.
on the bottom of the screen there should be a bar from left to right. That will display the sound level by default and the trimmed section of the sound in TRIM mode
middle will be used for the file selector in sound mode and maybe some cool graphics in other modes.
