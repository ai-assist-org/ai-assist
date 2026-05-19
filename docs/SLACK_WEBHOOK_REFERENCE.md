# Slack Webhook - Référence Rapide

## Configuration (.env)

```bash
# Webhook par défaut - utilisé pour logs/notes personnelles
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/XXX

# Webhook équipe - utilisé pour annonces/alertes équipe
SLACK_TEAM_WEBHOOK_URL=https://hooks.slack.com/services/T00/B01/YYY
```

## Utilisation en Langage Naturel

### Messages Personnels (channel par défaut)

L'agent utilise automatiquement le channel par défaut quand vous dites:

```
> Poste sur Slack: "..."
> Note sur Slack: "..."
> Log sur Slack: "..."
> Enregistre sur Slack: "..."
```

### Messages Équipe (channel team)

L'agent utilise le channel équipe quand vous mentionnez "équipe" ou "team":

```
> Alerte l'équipe sur Slack: "..."
> Annonce à l'équipe sur Slack: "..."
> Notifie l'équipe sur Slack: "..."
> Poste une annonce team sur Slack: "..."
> Envoie un message team sur Slack: "..."
```

## API Directe

```python
from ai_assist.slack_tools import SlackTools

slack = SlackTools()

# Channel par défaut (perso/logs)
await slack.execute_tool(
    "internal__post_slack_message",
    {"text": "Mon message"}
)

# Channel équipe
await slack.execute_tool(
    "internal__post_slack_message",
    {"text": "Annonce équipe", "channel": "team"}
)

# Avec formatage riche
await slack.execute_tool(
    "internal__post_slack_message",
    {
        "text": "Fallback text",
        "channel": "default",  # ou "team"
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Titre en gras*\n• Liste\n• D'items"
                }
            }
        ]
    }
)
```

## Paramètres de l'Outil

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `text` | string | ✅ Oui | Le texte du message (markdown Slack supporté) |
| `channel` | string | ❌ Non | `"default"` (défaut) ou `"team"` |
| `blocks` | array | ❌ Non | Blocs Block Kit pour formatage riche |

## Formatage Markdown Slack

| Syntaxe | Rendu |
|---------|-------|
| `*gras*` | **gras** |
| `_italique_` | _italique_ |
| `~barré~` | ~~barré~~ |
| `` `code` `` | `code` |
| `>quote` | Citation |
| `- item` | • item (liste) |
| `:emoji:` | 😀 (emoji) |
| `<https://url\|texte>` | [texte](https://url) |

## Exemples Courants

### Notification Simple
```
> Poste sur Slack: "Build terminé avec succès ✅"
```

### Alerte Équipe
```
> Alerte l'équipe: "Production déployée sur v2.1.0"
```

### Rapport Formaté
```
> Poste un rapport sur Slack avec:
  - Titre: **Déploiement Production**
  - Status: ✅ Succès
  - Version: v2.1.0
  - Durée: 5 minutes
```

### Intégration avec d'autres outils
```
> Vérifie les jobs DCI en erreur et poste un résumé sur mon channel Slack
> Cherche les tickets Jira bloqués et alerte l'équipe sur Slack
> Analyse les changements du KG et poste un rapport team sur Slack
```

## Retours de l'Outil

### Succès
```
✓ Message posted successfully to Slack (default channel)
✓ Message posted successfully to Slack (team channel)
```

### Erreurs Communes

| Erreur | Cause | Solution |
|--------|-------|----------|
| `SLACK_WEBHOOK_URL not configured` | Webhook par défaut absent | Ajouter dans `.env` |
| `SLACK_TEAM_WEBHOOK_URL not configured` | Webhook équipe demandé mais absent | Ajouter dans `.env` ou utiliser channel par défaut |
| `'text' parameter is required` | Paramètre text manquant | Fournir le texte du message |
| `Slack API error: 400` | Format du message invalide | Vérifier la syntaxe des blocks |
| `Timeout while posting` | Problème réseau | Vérifier connexion internet |

## Tests

```bash
# Lancer tous les tests (14 tests)
uv run pytest tests/test_slack_tools.py -v

# Tester les webhooks configurés
python scripts/test_slack_webhook.py

# Tester avec un message custom
python scripts/test_slack_webhook.py "Mon message de test"
```

## Dépannage Rapide

**L'outil n'apparaît pas:**
1. Vérifier que `.env` contient au moins un webhook
2. Relancer ai-assist
3. Chercher `✓ Added 1 Slack webhook tools` au démarrage

**Le message n'arrive pas:**
1. Vérifier l'URL du webhook dans l'app Slack
2. Vérifier que l'app est installée dans le workspace
3. Tester avec le script: `python scripts/test_slack_webhook.py`

**Mauvais channel utilisé:**
- Utilisez explicitement "équipe" ou "team" dans votre demande
- Ou spécifiez `channel="team"` en API directe

---

📖 **Guide complet:** [SLACK_WEBHOOK_SETUP.md](./SLACK_WEBHOOK_SETUP.md)
🚀 **Quick start:** [SLACK_WEBHOOK_QUICKSTART.md](./SLACK_WEBHOOK_QUICKSTART.md)
