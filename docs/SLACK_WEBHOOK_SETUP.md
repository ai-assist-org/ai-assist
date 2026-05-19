# Slack Webhook Integration - Guide d'Installation

Ce guide explique comment configurer et utiliser l'intégration Slack webhook dans ai-assist pour poster des messages automatiquement.

## Qu'est-ce qu'un Webhook Slack?

Un webhook Slack est une URL unique qui permet de poster des messages dans un channel Slack spécifique. C'est la méthode la plus simple pour envoyer des notifications à Slack.

**Avantages:**
- ✅ Configuration ultra-simple (2 minutes)
- ✅ Aucune dépendance supplémentaire
- ✅ Pas de serveur requis
- ✅ Intégration native avec l'agent AI
- ✅ Support de plusieurs webhooks (perso + équipe)

**Limitations:**
- ❌ Unidirectionnel (poster uniquement, pas de lecture)
- ❌ Un webhook = un channel fixe

## Installation - Étape par Étape

### 1. Créer un Webhook Slack (2 minutes)

1. Allez sur https://api.slack.com/apps
2. Cliquez sur **"Create New App"** → **"From scratch"**
3. Donnez un nom (ex: `ai-assist-webhook`) et sélectionnez votre workspace
4. Dans le menu gauche, cliquez sur **"Incoming Webhooks"**
5. Activez **"Activate Incoming Webhooks"**
6. Cliquez sur **"Add New Webhook to Workspace"**
7. Choisissez le channel où poster les messages (ex: `#general`, `#alerts`, etc.)
8. Autorisez l'application
9. **Copiez l'URL du webhook** (commence par `https://hooks.slack.com/services/...`)

### 2. Configurer ai-assist

Éditez votre fichier `.env` dans le projet ai-assist:

```bash
# Webhook par défaut (channel personnel/logs) - utilisé par défaut
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX

# Webhook équipe (optionnel) - pour les annonces/diffusion large
SLACK_TEAM_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/YYYYYYYYYYYYYYYYYYYY
```

**Configuration recommandée:**
- `SLACK_WEBHOOK_URL` → votre channel perso ou `#logs` (utilisé par défaut)
- `SLACK_TEAM_WEBHOOK_URL` → channel d'équipe type `#general` ou `#announcements` (optionnel)

**⚠️ SÉCURITÉ:**
- Ne committez JAMAIS le fichier `.env` dans Git
- L'URL du webhook doit rester secrète (elle permet de poster dans votre Slack!)
- Le `.gitignore` exclut déjà `.env` par défaut

### 3. Vérifier l'installation

Lancez ai-assist pour vérifier que l'outil est bien chargé:

```bash
uv run ai-assist
```

Vous devriez voir dans la sortie:
```
✓ Added 1 Slack webhook tools
```

## Utilisation

### Dans ai-assist (Interactive Mode)

Une fois configuré, vous pouvez demander à l'agent de poster sur Slack en langage naturel:

**Messages personnels (channel par défaut):**
```
> Poste un message sur Slack pour dire que le déploiement est terminé
> Log sur Slack: "Analyse terminée, 5 anomalies détectées"
> Note sur Slack: "CILAB-456 nécessite un suivi demain"
```

**Messages à l'équipe (channel team):**
```
> Envoie une alerte à l'équipe sur Slack: "CILAB-456 est bloqué"
> Notifie l'équipe sur Slack que les tests passent maintenant
> Poste une annonce team sur Slack: "Déploiement production terminé ✅"
```

L'agent choisit automatiquement le bon webhook selon le contexte (mots-clés: "équipe", "team", "annonce", "alerte équipe", etc.).

### Exemples Avancés

**Message simple:**
```
> Poste sur Slack: "Build terminé avec succès ✅"
```

**Message avec formatage Markdown:**
```
> Poste sur Slack un rapport avec:
  - Titre en gras: **Déploiement Production**
  - État: Succès
  - Commit: abc123
  - Durée: 5 minutes
```

**Intégration avec d'autres outils:**
```
> Vérifie les jobs DCI en erreur et envoie un résumé sur Slack
> Cherche les tickets Jira bloqués et poste une alerte Slack
> Résume les changements de la dernière heure dans le KG et poste sur Slack
```

### Utilisation Directe (API)

Si vous développez des scripts ou des compétences (skills), vous pouvez utiliser l'outil directement:

```python
# Dans un script Python
import asyncio
from ai_assist.slack_tools import SlackTools

slack = SlackTools()

# Message personnel (défaut)
await slack.execute_tool(
    "internal__post_slack_message",
    {"text": "Hello from Python!"}
)

# Message à l'équipe
await slack.execute_tool(
    "internal__post_slack_message",
    {"text": "Déploiement terminé!", "channel": "team"}
)

# Message avec formatage riche
await slack.execute_tool(
    "internal__post_slack_message",
    {
        "text": "Deployment terminé",
        "channel": "team",  # ou "default"
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Deployment Production*\n✅ Succès\nCommit: abc123"
                }
            }
        ]
    }
)
```

## Formatage des Messages

Slack supporte le formatage Markdown:

- `*gras*` → **gras**
- `_italique_` → _italique_
- `~barré~` → ~~barré~~
- `` `code` `` → `code`
- Listes avec `-` ou `•`
- Liens: `<https://example.com|Texte du lien>`
- Emojis: `:white_check_mark:` → ✅

Pour des messages plus complexes, utilisez [Block Kit](https://api.slack.com/block-kit).

## Configuration des Deux Webhooks

### Channel Personnel/Logs (SLACK_WEBHOOK_URL)

Créez un webhook pointant vers:
- Votre channel personnel/privé, OU
- Un channel `#logs` ou `#ai-logs` dédié

**Utilisé pour:**
- Notes personnelles
- Logs d'opérations
- Rappels et suivi
- Tests et debug

### Channel Équipe (SLACK_TEAM_WEBHOOK_URL)

Créez un second webhook pointant vers:
- `#general` ou `#team`
- `#announcements`
- Un channel partagé avec toute l'équipe

**Utilisé pour:**
- Alertes importantes
- Annonces d'équipe
- Notifications de déploiement
- Rapports partagés

### Créer un Webhook Supplémentaire

1. Dans votre app Slack, allez dans **"Incoming Webhooks"**
2. Cliquez sur **"Add New Webhook to Workspace"**
3. Choisissez le channel (ex: `#general` pour l'équipe)
4. Copiez l'URL et ajoutez-la dans `.env` comme `SLACK_TEAM_WEBHOOK_URL`

## Dépannage

### L'outil Slack n'apparaît pas

**Cause:** `SLACK_WEBHOOK_URL` n'est pas défini dans `.env`

**Solution:**
1. Vérifiez que `.env` existe dans `/Users/olivier/Dev/ai-assist/`
2. Vérifiez que la ligne `SLACK_WEBHOOK_URL=...` est présente et non commentée
3. Relancez ai-assist

### Erreur "Error: Slack API error: 400 - invalid_payload"

**Cause:** Le format du message n'est pas valide

**Solution:**
- Vérifiez la syntaxe des blocks si vous en utilisez
- Testez avec un message simple d'abord: `{"text": "Test"}`

### Erreur "Error: SLACK_TEAM_WEBHOOK_URL not configured"

**Cause:** Vous avez demandé à poster sur le channel équipe mais seul le webhook par défaut est configuré

**Solution:**
- Configurez `SLACK_TEAM_WEBHOOK_URL` dans `.env`, OU
- N'utilisez pas le paramètre `channel="team"` (utilisera le channel par défaut)

### Erreur "Error: Timeout while posting to Slack"

**Cause:** Problème réseau ou Slack indisponible

**Solution:**
- Vérifiez votre connexion internet
- Vérifiez le statut de Slack: https://status.slack.com/
- Le timeout est de 10 secondes par défaut

### L'URL du webhook ne fonctionne plus

**Cause:** Le webhook a été révoqué ou l'app désinstallée

**Solution:**
1. Allez sur https://api.slack.com/apps
2. Sélectionnez votre app
3. Vérifiez que l'app est installée dans le workspace
4. Recréez un webhook si nécessaire

## Tests

Lancez les tests pour vérifier que l'intégration fonctionne:

```bash
uv run pytest tests/test_slack_tools.py -v
```

Tous les tests doivent passer (14 tests, incluant les tests pour les 2 webhooks).

## Prochaines Étapes

Cette intégration webhook est parfaite pour poster des messages. Si vous avez besoin de **lire** les messages Slack ou d'interactions bidirectionnelles, considérez:

1. **Option simple:** Installer le serveur MCP Slack officiel (lecture + écriture)
2. **Option avancée:** Implémenter le bot Slack complet selon `docs/SLACK_BOT_PLAN.md`

## Ressources

- [Documentation Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Block Kit Builder](https://app.slack.com/block-kit-builder) - Créer des messages riches
- [Slack API Documentation](https://api.slack.com/)

---

**Créé le:** 2026-05-19
**Version ai-assist:** 0.1.0
