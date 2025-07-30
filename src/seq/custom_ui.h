#ifndef __SEQ_CUSTOM_UI_H
#define __SEQ_CUSTOM_UI_H

// intended to additionally include in sequence
// (besides event.h seq.h helpers.h)
// we do not allow _Complex and bool here

// DO NOT CHANGE THIS HEADER !

#include "misc/dllspec.h"
#include "misc/cppwrap.h"


enum checkbox {

	CHECKBOX_OFF = 1,
	CHECKBOX_ON
};


struct selection_opt {

	int id;
	const char* label;
};

struct seq_ui_selection {

	const char* tag;
	int id;
	const char* label;
	int opts_size;
	const struct selection_opt* opts;
	int val_default;
	const char* tooltip;
};

struct seq_ui_long { //also checkbox

	const char* tag;
	int id;
	const char* label;
	long limit[4]; // { min, max, inc, default }
	const char* tooltip;
	const char* unit;
};

struct seq_ui_double {

	const char* tag;
	int id;
	const char* label;
	double limit[4]; // { min, max, inc, default }
	const char* tooltip;
	const char* unit;
};


enum custom_ui_type {

	SELECTION = 0,
	BOOL,
	LONG,
	longarr,
	DOUBLE,
	doublearr,
};

struct custom_ui {

	int sizes[6];
	struct seq_ui_selection* selections;
	struct seq_ui_long* checkboxes;
	struct seq_ui_long* longs;
	struct seq_ui_long* longarr;
	struct seq_ui_double* doubles;
	struct seq_ui_double* doublearr;
};

BARTLIB_API struct custom_ui* BARTLIB_CALL seq_custom_ui_init(void);
BARTLIB_API void BARTLIB_CALL seq_custom_ui_free(struct custom_ui* ui);


// FIXME: workaround for sequence access 
#define _cil_pe_mode 0
#define _cil_contrast 1 // some UI bssfp behaviour
#define _cil_sms_distance 3 // write sms distance here
#define _cil_tiny 4

#define _cid_dummy 1 // workaround for commandline interface

#include "misc/cppwrap.h"

#endif // __SEQ_CUSTOM_UI_H
