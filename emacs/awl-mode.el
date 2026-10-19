;;; awl-mode.el --- Major mode for Agent Workflow Language files -*- lexical-binding: t; -*-

(defgroup awl nil
  "Major mode for AWL (Agent Workflow Language) files."
  :group 'languages)

(defvar awl-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?_ "w" table)
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\' "\"" table)
    table))

(defconst awl-task-fields
  '("Goal" "Context" "Constraints" "Success" "Expose"))

(defconst awl-font-lock-keywords
  `(;; Comments - full line and inline
    ("^[ \t]*\\(#.*\\)" 1 font-lock-comment-face t)
    ("\\( #.*\\)" 1 font-lock-comment-face t)
    ;; Directives - match @word at start of line (with optional indentation)
    ("^[ \t]*\\(@start\\|@end\\|@task\\|@set\\|@if\\|@else\\|@loop\\|@return\\|@goal\\|@fail\\|@wait\\|@while\\|@notify\\)\\b" 1 font-lock-keyword-face t)
    ;; Hints - match @hint after whitespace
    ("[ \t]\\(@no-history\\|@no-kg\\|@continue-on-failure\\)\\b" 1 font-lock-constant-face t)
    ;; Task/Goal ID after @task or @goal
    ("@task[ \t]+\\(\\w+\\)" 1 font-lock-function-name-face t)
    ("@goal[ \t]+\\(\\w+\\)" 1 font-lock-function-name-face t)
    ;; Task fields
    (,(concat "^[ \t]*\\(" (regexp-opt awl-task-fields) "\\):") 1 font-lock-type-face)
    ;; Variable interpolation ${...}
    ("\\${\\([^}]+\\)}" 0 font-lock-variable-name-face t)
    ;; @set variable name
    ("@set[ \t]+\\(\\w+\\)" 1 font-lock-variable-name-face t)
    ;; Task/workflow options: model=level effort=level, plus loop limits
    ("\\b\\(model\\|effort\\|limit\\|collect\\|max_actions\\|max_tool_calls\\|max_time\\|max_steps\\)=" 1 font-lock-builtin-face)
    ;; Loop variable: as <var>
    ("\\bas[ \t]+\\(\\w+\\)" 1 font-lock-variable-name-face)
    ;; Built-in functions
    ("\\b\\(len\\|not\\)\\b" 1 font-lock-builtin-face)))

(defun awl-indent-line ()
  "Indent current line for AWL mode."
  (interactive)
  (let ((indent 0)
        (cur-line-start (line-beginning-position)))
    (save-excursion
      ;; Scan all lines before current to compute nesting depth
      (goto-char (point-min))
      (while (< (point) cur-line-start)
        (let ((l (string-trim (thing-at-point 'line t))))
          (when (and l (string-match-p "^@\\(start\\|task\\|if\\|else\\|loop\\|goal\\)\\b" l))
            (setq indent (+ indent 2)))
          (when (and l (or (string-match-p "^@end\\b" l)
                           (string-match-p "^@else\\b" l)))
            (setq indent (max 0 (- indent 2)))))
        (forward-line 1)))
    ;; Dedent closing/branch directives
    (let ((line (string-trim (thing-at-point 'line t))))
      (when (and line (or (string-match-p "^@end\\b" line)
                          (string-match-p "^@else\\b" line)))
        (setq indent (max 0 (- indent 2)))))
    (indent-line-to indent)))

;;;###autoload
(define-derived-mode awl-mode prog-mode "AWL"
  "Major mode for editing AWL (Agent Workflow Language) files."
  :syntax-table awl-mode-syntax-table
  (setq-local font-lock-defaults '(awl-font-lock-keywords t))
  (setq-local indent-line-function #'awl-indent-line)
  (setq-local indent-tabs-mode nil)
  (setq-local tab-width 2)
  (setq-local comment-start "# ")
  (setq-local comment-end ""))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.awl\\'" . awl-mode))

(provide 'awl-mode)

;;; awl-mode.el ends here
