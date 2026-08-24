# Creating Personal Skills

This guide shows you how to create and manage local Agent Skills for your personal use.

## Quick Start

### 1. Create Your Skills Directory

```bash
mkdir -p ~/skills-personal
```

### 2. Create a New Skill

Use the helper script:

```bash
# From ai-assist repository
./scripts/create-skill ~/skills-personal home-automation "Control smart home devices"

# Or if ai-assist is installed
python -m ai_assist.scripts.create-skill ~/skills-personal meal-planning
```

### 3. Customize the Skill

Edit the generated `SKILL.md`:

```bash
cd ~/skills-personal/home-automation
$EDITOR SKILL.md
```

### 4. Install the Skill

```bash
ai-assist /interactive
/skill/install ~/skills-personal/home-automation@main
```

Changes auto-reload when you edit the skill file!

## Skill Structure

Each skill is a directory containing:

```
my-skill/
├── SKILL.md          # Required: Main skill definition
├── README.md         # Optional: Documentation
├── scripts/          # Optional: Executable scripts
├── references/       # Optional: Context documents
└── assets/           # Optional: Images, data files
```

## SKILL.md Format

The `SKILL.md` file has two parts:

### 1. YAML Frontmatter (Required)

```yaml
---
name: my-skill
description: Brief description of what this skill does
license: MIT
compatibility: ai-assist
---
```

**Fields:**
- `name` - Skill name (lowercase, hyphens, must match directory name)
- `description` - Brief description (1-1024 chars)
- `license` - Optional license identifier
- `compatibility` - Optional compatibility note
- `user-invocable` - Optional (default `true`). Set to `false` to hide the skill
  from the `/<skill-name>` command (it stays model-invocable).
- `argument-hint` - Optional hint shown in tab-completion, e.g. `[env] [version]`

### 2. Markdown Body (Your Instructions)

This is where you tell the AI how to behave with this skill:

```markdown
# My Skill

You are an AI assistant with knowledge about [topic].

## Instructions

- Guideline 1
- Guideline 2
- Preference or constraint

## Examples

### Example Usage

**User**: How do I...?

**Assistant**: You can do this by...

## References

- Link to docs
- Personal preferences
```

## Invoking a Skill as a Command

Beyond being picked up automatically by the assistant, any installed skill can
be run directly in interactive mode by typing `/<skill-name>`:

```
> /home-automation turn on the living room lights
```

This runs the skill's Markdown body as a query. Arguments after the name are
substituted into the body:

- `$ARGUMENTS` - the full argument string, verbatim (quotes preserved)
- `$1`, `$2`, ... - positional arguments; quoted segments stay grouped, so
  `/deploy "us east" v2` gives `$1` = `us east` and `$2` = `v2`
- If the body contains no placeholder, the arguments are appended as an
  `Arguments:` line.

Notes:
- Built-in commands (e.g. `/status`, `/help`) take precedence over a skill with
  the same name.
- Set `user-invocable: false` in the frontmatter to disable the command while
  keeping the skill available to the assistant.
- Type `/` and press Tab to see installed skill commands; `/help` lists them too.

## Example: Home Automation Skill

```yaml
---
name: home-automation
description: Control and monitor smart home devices
license: MIT
compatibility: ai-assist
---

# Home Automation

You are an AI assistant helping with home automation tasks.

## Devices

### Living Room
- **Lights**: Philips Hue (3 bulbs)
- **TV**: Samsung Smart TV
- **Thermostat**: Nest (target 68-72°F)

### Bedroom
- **Lights**: IKEA Tradfri (2 bulbs)
- **Fan**: Smart ceiling fan

## Instructions

1. When controlling devices:
   - Confirm actions before executing
   - Use natural language for commands
   - Consider time of day (don't turn on bright lights at night)

2. For automation suggestions:
   - Focus on energy efficiency
   - Prioritize comfort and convenience
   - Keep automations simple and reliable

## Examples

### Turn On Lights

**User**: Turn on the living room lights

**Assistant**: I'll turn on the 3 Philips Hue bulbs in the living room.
Since it's evening, I'll set them to 60% brightness with a warm white
color temperature. Would you like me to proceed?

### Check Temperature

**User**: What's the temperature?

**Assistant**: The Nest thermostat shows the current temperature is 70°F,
which is within your preferred range (68-72°F). The target is currently
set to 70°F. Would you like to adjust it?

## References

- Home Assistant API: http://homeassistant.local:8123
- Device manual: ~/Documents/smart-home-devices.pdf
```

## Example: Meal Planning Skill

```yaml
---
name: meal-planning
description: Help plan healthy meals based on dietary preferences
license: MIT
compatibility: ai-assist
---

# Meal Planning

You are an AI assistant helping with meal planning and nutrition.

## Dietary Preferences

- **Diet**: Mostly plant-based, occasional fish
- **Allergies**: None
- **Dislikes**: Cilantro, raw onions
- **Cuisine**: Mediterranean, Asian, Mexican
- **Cooking time**: Prefer 30-45 minutes max

## Pantry Staples

**Always available**:
- Rice, quinoa, pasta
- Canned beans, lentils
- Olive oil, soy sauce, spices
- Frozen vegetables

**Weekly shopping**:
- Fresh vegetables (seasonal)
- Tofu or tempeh
- Fresh herbs
- Bread, tortillas

## Instructions

1. When suggesting meals:
   - Focus on seasonal, whole foods
   - Balance protein, carbs, and vegetables
   - Keep prep time reasonable
   - Consider leftovers for lunch

2. For meal plans:
   - Generate 5-day plans (Mon-Fri)
   - Include simple breakfasts
   - Suggest 2-3 dinner options per day
   - Account for leftovers

## Examples

### Quick Dinner Suggestion

**User**: What should I make for dinner tonight?

**Assistant**: Based on your preferences, here are a few 30-minute options:

1. **Mediterranean Bowl**: Quinoa, roasted chickpeas, cucumber, tomatoes,
   tahini dressing
2. **Stir-fry**: Tofu and mixed vegetables over rice with ginger-soy sauce
3. **Tacos**: Black bean and sweet potato filling with avocado

Which sounds good?

### Weekly Plan

**User**: Plan meals for next week

**Assistant**: Here's a plant-based meal plan for Mon-Fri:

**Monday**: Lentil soup with whole grain bread
**Tuesday**: Thai basil tofu stir-fry
**Wednesday**: Chickpea curry with rice (make extra!)
**Thursday**: Curry leftovers
**Friday**: Veggie burrito bowls

Shopping list:
- Lentils, chickpeas, tofu
- Bell peppers, broccoli, sweet potatoes
- Coconut milk, curry paste
- Fresh basil, lime

## References

- Favorite recipes: ~/Documents/recipes/
- Seasonal produce guide: attached
```

## Example: Note-Taking Skill

```yaml
---
name: personal-notes
description: Help organize and manage personal notes and ideas
license: MIT
compatibility: ai-assist
---

# Personal Notes

You help me organize thoughts, ideas, and notes efficiently.

## Note Organization

**Directory Structure**:
```
~/notes/
├── inbox/          # Quick captures
├── projects/       # Active projects
├── reference/      # Permanent notes
├── archive/        # Completed items
└── journal/        # Daily journal entries
```

**File Format**: Markdown with YAML frontmatter

## Instructions

1. When creating notes:
   - Use clear, descriptive filenames (lowercase, hyphens)
   - Add YAML frontmatter with date, tags, and status
   - Use markdown headings for structure
   - Link related notes with `[[note-name]]` syntax

2. For organization:
   - Start new notes in `inbox/`
   - Process inbox weekly
   - Move to appropriate folder when categorized
   - Add to-do items use `- [ ]` checkboxes

3. Daily journal:
   - Create entries in `journal/YYYY-MM-DD.md`
   - Include: accomplishments, challenges, tomorrow's plan
   - Use simple, stream-of-consciousness style

## Templates

### Project Note

```markdown
---
date: 2024-01-15
tags: [project, active]
status: in-progress
---

# Project Name

## Goal

What I want to achieve

## Next Steps

- [ ] Task 1
- [ ] Task 2

## Notes

[Thoughts and ideas]

## Related
- [[other-note]]
```

### Quick Capture

```markdown
---
date: 2024-01-15
tags: [idea]
status: inbox
---

# Quick Idea

[Write down the idea quickly]

## Follow-up
- [ ] Review this
```

## Examples

### Capture an Idea

**User**: Save this idea: Build a script to auto-organize photos by date

**Assistant**: I'll create a note in your inbox:

```markdown
---
date: 2024-01-15
tags: [idea, automation]
status: inbox
---

# Auto-Organize Photos Script

Build a script to automatically organize photos by date taken.

## Next Steps
- [ ] Research Python libraries (PIL, exifread)
- [ ] Design folder structure (YYYY/MM/DD/)
- [ ] Handle duplicates

## Related
- [[photo-management]]
```

Saved to: `~/notes/inbox/auto-organize-photos.md`

## References

- Note templates: ~/notes/_templates/
- Vim shortcuts: Ctrl+] to follow links
```

## Managing Personal Skills

### Install a Skill

```bash
ai-assist /interactive
/skill/install ~/skills-personal/my-skill@main
```

### Update a Skill

Just edit the `SKILL.md` file - changes auto-reload:

```bash
cd ~/skills-personal/my-skill
vim SKILL.md
# Changes apply automatically!
```

### List Installed Skills

```bash
ai-assist /interactive
/skill/list
```

### Uninstall a Skill

```bash
ai-assist /interactive
/skill/uninstall my-skill
```

### Reinstall After Major Changes

If auto-reload doesn't pick up changes:

```bash
/skill/uninstall my-skill
/skill/install ~/skills-personal/my-skill@main
```

## Advanced Features

### Scripts Directory

Add executable scripts that AI can reference:

```bash
~/skills-personal/my-skill/
├── SKILL.md
└── scripts/
    ├── backup.sh
    └── deploy.py
```

**Important**: Script execution requires enabling in config:
```bash
export AI_ASSIST_ALLOW_SCRIPT_EXECUTION=true
```

See [SECURITY.md](../SECURITY.md) for details.

### References Directory

Add text files for additional context:

```bash
~/skills-personal/my-skill/
├── SKILL.md
└── references/
    ├── api-docs.md
    └── examples.txt
```

These are loaded and available to the AI.

### Assets Directory

Add data files, images, etc.:

```bash
~/skills-personal/my-skill/
├── SKILL.md
└── assets/
    ├── config.json
    └── diagram.png
```

## Tips for Effective Skills

### 1. Start Small

Begin with simple skills focused on one topic:
- ✅ `home-automation` - Smart home control
- ❌ `everything-personal` - Too broad

### 2. Be Specific

Include concrete examples and preferences:
```markdown
## Preferences
- Prefer TypeScript over JavaScript
- Use 2-space indentation
- Write tests first (TDD)
```

### 3. Add Context

Include your workflows and patterns:
```markdown
## My Workflow
1. Create feature branch from `develop`
2. Write tests first
3. Implement feature
4. Run `npm test && npm run lint`
5. Create PR with description
```

### 4. Use Examples

Show the AI how you want it to respond:
```markdown
### Example: Code Review

**User**: Review this code

**Assistant**: I'll review following your preferences:
- TypeScript types: ✅ All properly typed
- Tests: ⚠️ Missing test for edge case X
- Style: ✅ Follows 2-space indentation
...
```

### 5. Iterate

Update skills based on what works:
- If AI forgets something, add it to the skill
- If responses are too verbose, add verbosity guidance
- If AI makes wrong assumptions, clarify in examples

## Organizing Multiple Skills

### By Domain

```
~/skills-personal/
├── work/
│   ├── code-review/
│   ├── documentation/
│   └── deployment/
└── personal/
    ├── home-automation/
    ├── meal-planning/
    └── fitness/
```

### By Activity

```
~/skills-personal/
├── daily/
│   ├── morning-routine/
│   ├── work-planning/
│   └── evening-review/
├── weekly/
│   ├── meal-prep/
│   └── house-chores/
└── projects/
    ├── home-renovation/
    └── learning-spanish/
```

## Sharing Skills

To share a skill:

```bash
# 1. Copy to git repository
cp -r ~/skills-personal/my-skill ~/my-skills-repo/

# 2. Commit and push
cd ~/my-skills-repo
git add my-skill/
git commit -m "Add my-skill"
git push

# 3. Others can install
ai-assist /interactive
/skill/install github-user/my-skills-repo/my-skill@main
```

## Troubleshooting

### Skill Not Loading

Check the skill name matches directory:
```bash
# Directory name
ls ~/skills-personal/
# -> my-skill/

# Name in SKILL.md
grep "^name:" ~/skills-personal/my-skill/SKILL.md
# -> name: my-skill  ✅
```

### Changes Not Applying

1. Check installed-skills.json:
```bash
cat ~/.ai-assist/installed-skills.json
```

2. Reinstall if needed:
```bash
/skill/uninstall my-skill
/skill/install ~/skills-personal/my-skill@main
```

### Invalid Skill Format

Validate YAML frontmatter:
```bash
# Extract frontmatter and validate
python3 << 'EOF'
import yaml
with open('/home/flepied/skills-personal/my-skill/SKILL.md') as f:
    content = f.read()
    if content.startswith('---'):
        parts = content.split('---', 2)
        metadata = yaml.safe_load(parts[1])
        print("✅ Valid YAML")
        print(metadata)
EOF
```

## Related Documentation

- **[README.md](../README.md)** - Main documentation
- **[SECURITY.md](../SECURITY.md)** - Script execution security
- **[agentskills.io](https://agentskills.io)** - Official specification
