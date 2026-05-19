# Slack Webhook - Quick Start (5 minutes)

Ajoutez l'intégration Slack à ai-assist en 3 étapes simples.

## 1. Obtenir les URLs des Webhooks (3 min)

1. Allez sur https://api.slack.com/apps
2. **Create New App** → **From scratch**
3. Nommez votre app (ex: `ai-assist-webhook`)
4. Menu gauche: **Incoming Webhooks** → **Activate Incoming Webhooks** (ON)

**Webhook 1 - Channel personnel/logs (requis):**
5. **Add New Webhook to Workspace**
6. Choisissez votre channel perso ou `#logs` → **Allow**
7. **Copiez l'URL du webhook #1**

**Webhook 2 - Channel équipe (optionnel mais recommandé):**
8. **Add New Webhook to Workspace** (encore)
9. Choisissez `#general` ou `#team` → **Allow**
10. **Copiez l'URL du webhook #2**

## 2. Configurer ai-assist (30 sec)

Ajoutez dans `.env`:

```bash
# Channel personnel/logs (par défaut)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX

# Channel équipe (optionnel - pour les annonces)
SLACK_TEAM_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/YYYYYYYYYYYYYYYYYYYY
```

## 3. Tester (30 sec)

```bash
# Test rapide
python scripts/test_slack_webhook.py

# Ou lancez ai-assist
uv run ai-assist
```

Dans ai-assist:
```
# Message perso (défaut)
> Poste sur Slack: "ai-assist est configuré! 🚀"

# Message équipe
> Annonce à l'équipe sur Slack: "ai-assist est opérationnel!"
```

✅ **C'est tout!** Vous pouvez maintenant poster sur Slack (perso ou équipe) via l'agent AI.

---

📖 **Guide complet:** Voir [SLACK_WEBHOOK_SETUP.md](./SLACK_WEBHOOK_SETUP.md)
