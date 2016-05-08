
SECTIONS
{
	. = ALIGN(8);
	.utests_array :
	{
		_utests_begin = .;
    		KEEP( *(.utest))
		_utests_end = .;
	}
}
INSERT AFTER .rodata;

