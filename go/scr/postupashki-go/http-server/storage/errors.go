package storage

import "errors"

var (
	ErrNotFound         = errors.New("key not found")
	ErrKeyAlreadyExists = errors.New("key already exists")
	ErrTaskIsInProgress = errors.New("task is in progress")
	ErrInternalError    = errors.New("operation failed")
)
